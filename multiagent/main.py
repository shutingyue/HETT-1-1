import torch
torch.backends.cudnn.enabled = False
##关闭 CuDNN 加速器

import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
import sys, os

from defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 先加入绝对路径，否则会报错，注意__file__表示的是当前执行文件的路径
from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import (
    init_distributed,
    is_dist_avail_and_initialized,
    get_rank,
    get_world_size,
    is_main_process,
    barrier as dist_barrier,
    cleanup_distributed,
)
from utils.distributed import all_gather, merge_dist_results

from agent import NavCMTAgent
from env import CityNavBatch
from parser import parse_args

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def is_distributed():
    return is_dist_avail_and_initialized()


def distributed_barrier():
    dist_barrier()


def debug_ddp(args, message):
    if getattr(args, "debug_ddp", False):
        print(f"[DDP] rank={get_rank()} local_rank={args.local_rank} world_size={get_world_size()} {message}")


EXPERIMENT_CONFIG_GROUPS = [
    ("Experiment", [
        "seed",
        "mode",
        "world_size",
        "local_rank",
        "rank",
        "log_dir",
        "debug_ddp",
        "debug_anomaly",
    ]),
    ("Training", [
        "epochs",
        "batch_size",
        "learning_rate",
        "feedback",
        "eval_every",
        "save_every",
        "log_every",
        "use_progress_bar",
        "progress_log_interval",
        "resume_optimizer",
        "max_train_batches_per_epoch",
    ]),
    ("UMTI", [
        "use_umti",
        "use_memory_type_embedding",
        "num_memory_types",
        "debug_memory_tokens",
    ]),
    ("ELAM", [
        "use_elam",
        "debug_elam",
        "num_elam_roles",
        "elam_num_heads",
        "elam_dropout",
        "elam_fusion_mode",
        "elam_aux_weight",
        "metric_cell_loss_weight",
        "soft_spatial_loss_weight",
        "query_div_loss_weight",
        "soft_spatial_sigma",
    ]),
    ("UASC", [
        "use_uasc",
        "uasc_control_mode",
        "uasc_hidden_size",
        "uasc_dropout",
        "uasc_aux_dim",
        "uasc_lambda_conf",
        "uasc_lambda_stage",
        "uasc_lambda_stop",
        "use_uasc_calib",
        "uasc_lambda_calib",
        "uasc_stage_radius",
        "uasc_tau_conf",
        "uasc_tau_stage",
        "uasc_tau_stop",
        "uasc_debug",
    ]),
    ("Memory", [
        "grid_size",
        "enable_topo_memory",
        "persistent_topo_memory",
        "use_time_decay",
        "spatial_compression",
    ]),
    ("Topo", [
        "topo_max_nodes",
        "global_retrieve_k",
        "local_hops",
        "topo_knn",
        "topo_use_graph_encoder",
        "topo_message_passing_layers",
        "use_topo_gate",
    ]),
    ("Topo Retrieval Weights", [
        "retrieve_goal_weight",
        "retrieve_visual_weight",
        "retrieve_visit_weight",
        "goal_create_norm_threshold",
    ]),
    ("Reserved Future Modules", [
        "use_semantic_nodes",
        "max_semantic_nodes",
        "use_uncertainty_policy",
        "use_landmark_nodes",
        "use_event_nodes",
    ]),
]


def format_experiment_config(args):
    lines = ["Experiment configuration:"]
    for group_name, keys in EXPERIMENT_CONFIG_GROUPS:
        group_lines = [f"{key}: {getattr(args, key)}" for key in keys if hasattr(args, key)]
        if not group_lines:
            continue
        lines.append("")
        lines.append(f"[{group_name}]")
        lines.extend(group_lines)
    return "\n".join(lines)


def print_experiment_config(args):
    config_text = format_experiment_config(args)
    print(config_text)
    return config_text


def get_checkpoint_dir():
    return os.path.abspath(str(GOAL_PREDICTOR_CHECKPOINT_DIR))


def save_args_json(args, output_dir, filename="args.json"):
    os.makedirs(output_dir, exist_ok=True)
    args_path = os.path.join(output_dir, filename)
    with open(args_path, "w", encoding="utf-8") as outf:
        json.dump(vars(args), outf, indent=4, sort_keys=True, default=str)
    return args_path


def append_train_record(record_file, data, verbose=True):
    if not is_main_process():
        return
    if verbose:
        print(data)
    os.makedirs(os.path.dirname(record_file), exist_ok=True)
    with open(record_file, 'a', encoding='utf-8') as f:
        f.write(data)
        if not data.endswith('\n'):
            f.write('\n')


def format_compression_logs(logs):
    keys = [
        'tokens_before',
        'tokens_after',
        'near_tokens',
        'far_summary_tokens',
        'pruned_tokens',
        'merged_away_tokens',
    ]
    values = []
    for key in keys:
        mean_value = sum(logs[key]) / max(len(logs[key]), 1)
        values.append((key, mean_value))
    return "compression " + " ".join(f"{key} {value:.4f}" for key, value in values)


def format_elam_logs(logs):
    return (
        "ELAM_loss {elam:.4f} metric_cell {metric:.4f} soft_spatial {spatial:.4f} "
        "query_div {query:.4f} align_conf {conf:.4f} align_entropy {entropy:.4f}"
    ).format(
        elam=_mean_log_value(logs, 'elam_loss_mean'),
        metric=_mean_log_value(logs, 'metric_cell_loss_mean'),
        spatial=_mean_log_value(logs, 'soft_spatial_loss_mean'),
        query=_mean_log_value(logs, 'query_div_loss_mean'),
        conf=_mean_log_value(logs, 'alignment_confidence_mean'),
        entropy=_mean_log_value(logs, 'alignment_entropy_mean'),
    )


def format_uasc_logs(logs):
    return (
        "UASC_conf {conf:.4f} UASC_stage {stage:.4f} UASC_stop {stop:.4f} "
        "UASC_total {total:.4f} coarse_conf {mean_conf:.4f} "
        "stage_prob {mean_stage:.4f} stop_prob {mean_stop:.4f}"
    ).format(
        conf=_mean_log_value(logs, 'uasc_conf'),
        stage=_mean_log_value(logs, 'uasc_stage'),
        stop=_mean_log_value(logs, 'uasc_stop'),
        total=_mean_log_value(logs, 'uasc_total'),
        mean_conf=_mean_log_value(logs, 'uasc_mean_coarse_conf'),
        mean_stage=_mean_log_value(logs, 'uasc_mean_stage_prob'),
        mean_stop=_mean_log_value(logs, 'uasc_mean_stop_prob'),
    )


def _mean_log_value(logs, key, default=0.0):
    values = []
    for value in logs.get(key, []):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return default
    return sum(values) / max(len(values), 1)


def _min_log_value(logs, key, default=0.0):
    values = []
    for value in logs.get(key, []):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return min(values) if values else default


def _max_log_value(logs, key, default=0.0):
    values = []
    for value in logs.get(key, []):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return max(values) if values else default


def format_topo_logs(logs):
    if not logs.get('avg_place_nodes', []) and not logs.get('total_place_nodes', []):
        return "[topo_stats] disabled"
    avg_place_nodes = _mean_log_value(logs, 'avg_place_nodes', _mean_log_value(logs, 'total_place_nodes'))
    max_place_nodes = _mean_log_value(logs, 'max_place_nodes_used', avg_place_nodes)
    min_place_nodes = _mean_log_value(logs, 'min_place_nodes_used', avg_place_nodes)
    create_count = _mean_log_value(logs, 'create_place_nodes_count', _mean_log_value(logs, 'step_new_place_nodes'))
    stored_count = _mean_log_value(logs, 'num_place_nodes_stored', avg_place_nodes)
    update_count = _mean_log_value(logs, 'update_existing_place_nodes_count', _mean_log_value(logs, 'step_updated_place_nodes'))
    merge_count = _mean_log_value(logs, 'merge_place_nodes_count', _mean_log_value(logs, 'step_merged_place_nodes'))
    op_count = max(create_count + update_count, 1e-6)
    create_rate = _mean_log_value(logs, 'create_rate', create_count / op_count)
    update_rate = _mean_log_value(logs, 'update_rate', update_count / op_count)
    merge_rate = _mean_log_value(logs, 'merge_rate', merge_count / op_count)
    global_k = _mean_log_value(logs, 'global_retrieved_nodes')
    local_k = _mean_log_value(logs, 'local_retrieved_nodes')
    valid_topo_tokens = _mean_log_value(logs, 'num_valid_topo_tokens_to_umti', _mean_log_value(logs, 'topo_mask_sum'))
    topo_mask_sum = _mean_log_value(logs, 'topo_mask_sum')
    topo_norm_before_gate = _mean_log_value(logs, 'topo_token_norm_before_gate')
    topo_norm_after_gate = _mean_log_value(logs, 'topo_token_norm_after_gate')
    retrieval_coverage = _mean_log_value(logs, 'retrieval_coverage', global_k / max(avg_place_nodes, 1.0))
    goal_rel_avg = _mean_log_value(logs, 'goal_relevance', _mean_log_value(logs, 'avg_goal_relevance'))
    goal_rel_min = _min_log_value(logs, 'goal_relevance', goal_rel_avg)
    goal_rel_max = _max_log_value(logs, 'goal_relevance', _mean_log_value(logs, 'max_goal_relevance', goal_rel_avg))
    visual_change_avg = _mean_log_value(logs, 'visual_change')
    visual_change_min = _min_log_value(logs, 'visual_change', visual_change_avg)
    visual_change_max = _max_log_value(logs, 'visual_change', visual_change_avg)
    created_goal = _mean_log_value(
        logs,
        'created_goal_relevance',
        _mean_log_value(logs, 'goal_relevance_of_created_nodes', float('nan')),
    )
    updated_goal = _mean_log_value(
        logs,
        'updated_goal_relevance',
        _mean_log_value(logs, 'goal_relevance_of_updated_nodes', float('nan')),
    )
    merged_goal = _mean_log_value(
        logs,
        'merged_goal_relevance',
        _mean_log_value(logs, 'goal_relevance_of_merged_nodes', float('nan')),
    )
    return (
        "[topo_stats] place_nodes avg={:.2f} min={:.2f} max={:.2f} sat={:.3f} "
        "created={:.2f} stored={:.2f} update={:.2f} merge={:.2f} create_rate={:.3f} update_rate={:.3f} merge_rate={:.3f} "
        "global_k={:.2f} local_k={:.2f} valid_topo_tokens={:.2f} topo_mask_sum={:.2f} "
        "coverage={:.3f} active_valid={:.3f} empty={:.3f} "
        "goal_rel avg={:.4f} range=[{:.4f},{:.4f}] created_goal={:.4f} updated_goal={:.4f} merged_goal={:.4f} "
        "visual_change avg={:.4f} range=[{:.4f},{:.4f}] "
        "token_norm topo={:.4f}+/-{:.4f} global={:.4f} local={:.4f} before_gate={:.4f} after_gate={:.4f}"
    ).format(
        avg_place_nodes,
        min_place_nodes,
        max_place_nodes,
        _mean_log_value(logs, 'node_saturation_ratio'),
        create_count,
        stored_count,
        update_count,
        merge_count,
        create_rate,
        update_rate,
        merge_rate,
        global_k,
        local_k,
        valid_topo_tokens,
        topo_mask_sum,
        retrieval_coverage,
        _mean_log_value(logs, 'active_node_valid_ratio'),
        _mean_log_value(logs, 'empty_retrieval_ratio'),
        goal_rel_avg,
        goal_rel_min,
        goal_rel_max,
        created_goal,
        updated_goal,
        merged_goal,
        visual_change_avg,
        visual_change_min,
        visual_change_max,
        _mean_log_value(logs, 'topo_token_norm_mean'),
        _mean_log_value(logs, 'topo_token_norm_std'),
        _mean_log_value(logs, 'global_token_norm_mean'),
        _mean_log_value(logs, 'local_token_norm_mean'),
        topo_norm_before_gate,
        topo_norm_after_gate,
    )


def get_tokenizer(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/cver/xcding/code/tokenizer_files/bert-base-uncase')
    return tokenizer


def build_train_dataset(args, rank=0):
    # tok = get_tokenizer(args)
    # print(rank)
    dataset_class = CityNavBatch

    train_env = dataset_class(
        'train_seen',
        args,
        batch_size=args.batch_size,
        seed=args.seed + rank,
        rank=rank,
        world_size=args.world_size
    )

    val_env_names = ['val_seen', 'val_unseen', ]  # 'test_unseen'
    # val_env_names = ['val_seen',]  # 'test_unseen'

    val_envs = {}
    if is_main_process():
        for split in val_env_names:
            val_env = dataset_class(
                split, args,
                batch_size=args.batch_size,
                seed=args.seed + rank,
                rank=rank,
                world_size=1
            )

            val_envs[split] = val_env

    return train_env, val_envs


def build_val_dataset(args, rank=0):
    # tok = get_tokenizer(args)
    # print(rank)
    dataset_class = CityNavBatch

    val_env_names = ['val_seen', 'val_unseen', 'test_unseen', ]  # 'test_unseen'
    # val_env_names = ['visualization' ]  # 'test_unseen'

    val_envs = {}
    if is_main_process():
        for split in val_env_names:
            val_env = dataset_class(
                split, args,
                batch_size=args.batch_size,
                seed=args.seed + rank,
                rank=rank,
                world_size=1
            )

            val_envs[split] = val_env

    return val_envs

def build_vis_dataset(args, rank=0):
    # tok = get_tokenizer(args)
    # print(rank)
    dataset_class = CityNavBatch
    #
    # val_env_names = ['val_seen', 'val_unseen', 'test_unseen', ]  # 'test_unseen'
    val_env_names = ['visualization' ]  # 'test_unseen'

    val_envs = {}
    if is_main_process():
        for split in val_env_names:
            val_env = dataset_class(
                split, args,
                batch_size=args.batch_size,
                seed=args.seed + rank,
                rank=rank,
                world_size=1
            )

            val_envs[split] = val_env

    return val_envs


def train(args, train_env, val_envs, rank=-1):
    # print('?')
    default_gpu = is_main_process()

    if default_gpu:
        checkpoint_dir = get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_text = print_experiment_config(args)
        args_path = save_args_json(args, checkpoint_dir, "args.json")
        training_args_path = save_args_json(args, checkpoint_dir, "training_args.json")
        record_file = os.path.join(checkpoint_dir, 'train.txt')
        valid_record_file = os.path.join(checkpoint_dir, 'valid.txt')
        print(f"[SaveDir] checkpoint_dir: {checkpoint_dir}")
        print(f"[SaveDir] train_record_file: {record_file}")
        print(f"[SaveDir] valid_record_file: {valid_record_file}")
        print(f"[SaveDir] args_json: {args_path}")
        print(f"[Args] saved to {args_path}")
        print(f"[Args] training args saved to {training_args_path}")
        # writer = SummaryWriter(log_dir=args.log_dir)
        append_train_record(record_file, config_text + '\n', verbose=False)
        append_train_record(record_file, f"[Args] saved to {args_path}\n", verbose=False)
        append_train_record(record_file, f"[Args] training args saved to {training_args_path}\n", verbose=False)
        train_batches_est = int(np.ceil(float(train_env.size()) / float(train_env.batch_size)))
        print(
            "[DDP] train dataset shard size={} world_size={} batch_size={} estimated num batches={} max_train_batches_per_epoch={}".format(
                train_env.size(),
                get_world_size(),
                args.batch_size,
                train_batches_est,
                args.max_train_batches_per_epoch,
            )
        )

    best_val = {'val_unseen': {"sr": 0., "state": ""}, 'val_unseen_full_traj': {"sr": 0., "state": ""}}

    # first evaluation
    if args.eval_first:
        loss_str = ""
        debug_ddp(args, "entering eval_first barrier")
        distributed_barrier()
        if default_gpu:
            debug_ddp(args, "rank0 running eval_first")

            for env_name, env in val_envs.items():
                agent_class_eval = NavCMTAgent
                agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
                start_epoch = 0

                if args.checkpoint is not None:
                    start_epoch = agent_eval.load(os.path.join(args.checkpoint))
                    if default_gpu:
                        append_train_record(
                            record_file,
                            "\nLOAD the model from {}, epoch {}".format(args.checkpoint, start_epoch),
                        )

                agent_eval.env = env
                # sampler = DistributedSampler(env, num_replicas=args.world_size, rank=rank)
                loader = DataLoader(env, batch_size=1)
                # Get validation distance from goal under test evaluation conditions
                agent_eval.test(loader, env_name=env_name, feedback='student')
                pred_results = agent_eval.get_results()

                score_summary, result = env.eval_metrics(pred_results)
                loss_str += ", %s \n" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                if env_name in best_val:
                    if score_summary['sr'] >= best_val[env_name]['sr']:
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Epoch %d %s' % (start_epoch, loss_str)
            append_train_record(record_file, loss_str)
        debug_ddp(args, "leaving eval_first barrier")
        distributed_barrier()

    torch.cuda.empty_cache()
    agent_class = NavCMTAgent
    agent = agent_class(args, rank=rank)

    # resume file
    start_epoch = 0
    if args.checkpoint is not None:
        start_epoch = agent.load(os.path.join(args.checkpoint))
        if default_gpu:
            append_train_record(
                record_file,
                "\nLOAD the model from {}, epoch {}".format(args.checkpoint, start_epoch),
            )

    # Start Training
    start = time.time()
    if default_gpu:
        append_train_record(
            record_file,
            '\nListener training starts, start epoch: %s' % str(start_epoch)
        )



    torch.cuda.empty_cache()
    # interval = int(train_env.size() / args.batch_size) * args.log_every

    # zero_start_iter = 0
    for idx in range(start_epoch, args.epochs):
        agent.logs = defaultdict(list)

        # iter = idx + interval
        # if args.train_val_on_full:
        #     agent.env = train_full_traj_env
        # else:
        agent.env = train_env
        # print(agent.env.size())
        loader = DataLoader(agent.env, batch_size=1)
        # print(loader.dataset.size())

        max_train_batches = (
            int(args.max_train_batches_per_epoch)
            if getattr(args, 'max_train_batches_per_epoch', -1) is not None
            and int(args.max_train_batches_per_epoch) > 0
            else None
        )
        train_passes = 1 if max_train_batches is not None else args.log_every

        # Train before evaluating this outer epoch. In short-run mode the batch cap is
        # for this outer epoch, so avoid multiplying it by the legacy log_every passes.
        if getattr(args, "debug_ddp", False):
            sampler_name = type(getattr(loader, "sampler", None)).__name__
            debug_ddp(args, f"train_loader length=iterable sampler={sampler_name}")
        actual_train_batches = agent.train(loader, train_passes, feedback=args.feedback,
                                           nss_w_weighting=1,
                                           max_batches_per_epoch=max_train_batches,
                                           epoch_idx=idx)  # nss_w_weighting = max(0, (args.iters/2 - idx)/ (args.iters/2)))

        distributed_barrier()

        should_eval = (idx % max(int(args.eval_every), 1)) == 0
        should_save = (idx % max(int(args.save_every), 1)) == 0
        latest_path = os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest")

        if default_gpu:
            ml_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)

            direction_loss = sum(agent.logs['direction_loss']) / max(len(agent.logs['direction_loss']), 1)

            progress_loss = sum(agent.logs['progress_loss']) / max(len(agent.logs['progress_loss']), 1)
            goal_predict_loss = sum(agent.logs['goal_predict_loss']) / max(len(agent.logs['goal_predict_loss']), 1)
            # target_predict_loss = sum(agent.logs['target_predict_loss']) / max(len(agent.logs['target_predict_loss']), 1)
            # writer.add_scalar("loss/IL_loss", IL_loss, iter)

            train_loss_str = "\nepoch %d train loss IL_loss %.4f direction_loss %.4f progress_loss %.4f goal_predict_loss %.4f" % (
                idx, ml_loss, direction_loss, progress_loss, goal_predict_loss
            )
            append_train_record(record_file, train_loss_str)
            stage1_step = sum(agent.logs['stage1_step']) / max(len(agent.logs['stage1_step']), 1)
            stage2_step = sum(agent.logs['stage2_step']) / max(len(agent.logs['stage2_step']), 1)
            stage2_rotate = sum(agent.logs['stage2_rotate']) / max(len(agent.logs['stage2_rotate']), 1)

            append_train_record(
                record_file,
                "\nstage %.4f %.4f %.4f" % (
                    stage1_step, stage2_step, stage2_rotate)
            )
            append_train_record(record_file, "\n%s" % format_compression_logs(agent.logs))
            if args.use_elam:
                append_train_record(record_file, "\n%s" % format_elam_logs(agent.logs))
            if args.use_uasc:
                append_train_record(record_file, "\n%s" % format_uasc_logs(agent.logs))
            if args.enable_topo_memory:
                append_train_record(record_file, "\n%s" % format_topo_logs(agent.logs))
            append_train_record(record_file, "\nactual_train_batches_this_epoch %d" % actual_train_batches)
            print(f"[DDP] actual_train_batches_this_epoch={actual_train_batches}")

        debug_ddp(args, "entering checkpoint barrier")
        distributed_barrier()
        if default_gpu and (should_save or should_eval):
            saved_path = agent.save(idx, latest_path)
            debug_ddp(args, f"checkpoint save path={saved_path}")
            append_train_record(record_file, "\ncheckpoint save path: %s" % saved_path)
        elif default_gpu:
            print(f"[Checkpoint] save skipped at epoch {idx} (save_every={args.save_every}, eval_every={args.eval_every})")
            append_train_record(
                record_file,
                "\ncheckpoint save skipped at epoch %d (save_every=%s, eval_every=%s)" % (
                    idx, args.save_every, args.eval_every
                )
            )
        debug_ddp(args, "leaving checkpoint barrier")
        distributed_barrier()

        # Run validation on rank 0 only, with all other ranks waiting here.
        debug_ddp(args, "entering eval barrier")
        distributed_barrier()
        if default_gpu:
            if should_eval:
                debug_ddp(args, "rank0 running eval")
                loss_str = "\nepoch {}".format(idx)

                agent_class_eval = NavCMTAgent
                agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
                print("Loaded the listener model at epoch %d from %s" % \
                      (agent_eval.load(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest")),
                       os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest")))
                for env_name, env in val_envs.items():
                    agent_eval.logs = defaultdict(list)
                    agent_eval.env = env
                    loader = DataLoader(env, batch_size=1)
                    # Get validation distance from goal under test evaluation conditions
                    agent_eval.test(loader, env_name=env_name, feedback='student')
                    pred_results = agent_eval.get_results()

                    score_summary, result = env.eval_metrics(pred_results)
                    stage1_step = sum(agent_eval.logs['stage1_step']) / max(len(agent_eval.logs['stage1_step']), 1)
                    stage2_step = sum(agent_eval.logs['stage2_step']) / max(len(agent_eval.logs['stage2_step']), 1)
                    stage2_rotate = sum(agent_eval.logs['stage2_rotate']) / max(len(agent_eval.logs['stage2_rotate']), 1)

                    append_train_record(
                        record_file,
                        "\nstage %.4f %.4f %.4f" % (
                            stage1_step, stage2_step, stage2_rotate)
                    )
                    append_train_record(record_file, "\n%s" % format_compression_logs(agent_eval.logs))
                    if args.enable_topo_memory:
                        append_train_record(record_file, "\n%s" % format_topo_logs(agent_eval.logs))
                    loss_str += "\n%s " % env_name
                    for metric, val in score_summary.items():
                        loss_str += ', %s: %.2f' % (metric, val)
                        # writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
                    if env_name in best_val:
                        if score_summary['sr'] >= best_val[env_name]['sr']:
                            best_val[env_name]['sr'] = score_summary['sr']
                            best_val[env_name]['state'] = 'Epoch %d %s' % (idx, loss_str)
                            best_path = agent_eval.save(idx, os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "best_%s" % (env_name)))
                            append_train_record(record_file, "best checkpoint save path: %s" % best_path)

                append_train_record(
                    record_file,
                    ('\n%s (%d %d%%) %s' % (
                        timeSince(start, float(idx + 1) / args.epochs), idx + 1, float(idx + 1) / args.epochs * 100,
                        loss_str))
                )
                append_train_record(record_file, "BEST RESULT TILL NOW")
                for env_name in best_val:
                    append_train_record(record_file, env_name + ' | ' + best_val[env_name]['state'])
            else:
                print(f"[Eval] skipped at epoch {idx} (eval_every={args.eval_every})")
                append_train_record(record_file, "\nepoch %d val_seen metrics: skipped (eval_every=%s)" % (idx, args.eval_every))
                append_train_record(record_file, "epoch %d val_unseen metrics: skipped (eval_every=%s)" % (idx, args.eval_every))
                append_train_record(record_file, "BEST RESULT TILL NOW")
                for env_name in best_val:
                    append_train_record(record_file, env_name + ' | ' + best_val[env_name]['state'])
        debug_ddp(args, "leaving eval barrier")
        distributed_barrier()
        torch.cuda.empty_cache()


def valid(args, val_envs, rank=-1):
    default_gpu = is_main_process()
    if default_gpu:
        checkpoint_dir = get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        agent_class_eval = NavCMTAgent
        agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
        epoch = agent_eval.load(args.checkpoint)
        if args.checkpoint is not None:
            print("Loaded the listener model at epoch %d from %s" % \
                  (epoch, args.checkpoint))
            loss_str = "\nepoch {}".format(epoch)

        with open(os.path.join(checkpoint_dir, 'validation_args.json'), 'w', encoding='utf-8') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(checkpoint_dir, 'valid.txt')
        for env_name, env in val_envs.items():
            agent_eval.logs = defaultdict(list)
            agent_eval.env = env
            loader = DataLoader(env, batch_size=1)
            # Get validation distance from goal under test evaluation conditions
            agent_eval.test(loader, env_name=env_name, feedback='student')
            pred_results = agent_eval.get_results()

            score_summary, result = env.eval_metrics(pred_results)
            stage1_step = sum(agent_eval.logs['stage1_step']) / max(len(agent_eval.logs['stage1_step']), 1)
            stage2_step = sum(agent_eval.logs['stage2_step']) / max(len(agent_eval.logs['stage2_step']), 1)
            stage2_rotate = sum(agent_eval.logs['stage2_rotate']) / max(len(agent_eval.logs['stage2_rotate']), 1)

            write_to_record_file(
                "\nstage %.4f %.4f %.4f" % (
                    stage1_step, stage2_step, stage2_rotate),
                record_file
            )
            write_to_record_file("\n%s" % format_compression_logs(agent_eval.logs), record_file)
            if args.enable_topo_memory:
                write_to_record_file("\n%s" % format_topo_logs(agent_eval.logs), record_file)
            loss_str += "\n%s " % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
                # writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
        write_to_record_file(
            ('\n%s' % loss_str),
            record_file
        )
        # json.dump(
        #     result,
        #     open(os.path.join(args.pred_dir, "eval_detail_%s.json" % env_name), 'w'),
        #     sort_keys=True, indent=4, separators=(',', ': ')
        # )

def visualize(args, vis_envs, rank=-1):
    default_gpu = is_main_process()
    if default_gpu:
        checkpoint_dir = get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        agent_class_eval = NavCMTAgent
        agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
        epoch = agent_eval.load(args.checkpoint)
        if args.checkpoint is not None:
            print("Loaded the listener model at epoch %d from %s" % \
                  (epoch, args.checkpoint))
            loss_str = "\nepoch {}".format(epoch)

        with open(os.path.join(checkpoint_dir, 'validation_args.json'), 'w', encoding='utf-8') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(checkpoint_dir, 'valid.txt')
        for env_name, env in vis_envs.items():
            agent_eval.logs = defaultdict(list)
            agent_eval.env = env
            loader = DataLoader(env, batch_size=1)
            # Get validation distance from goal under test evaluation conditions
            agent_eval.visualize(loader, env_name=env_name, feedback='student')
            pred_results = agent_eval.get_results()

        #     score_summary, result = env.eval_metrics(pred_results)
        #     stage1_step = sum(agent_eval.logs['stage1_step']) / max(len(agent_eval.logs['stage1_step']), 1)
        #     stage2_step = sum(agent_eval.logs['stage2_step']) / max(len(agent_eval.logs['stage2_step']), 1)
        #     stage2_rotate = sum(agent_eval.logs['stage2_rotate']) / max(len(agent_eval.logs['stage2_rotate']), 1)
        #
        #     write_to_record_file(
        #         "\nstage %.4f %.4f %.4f" % (
        #             stage1_step, stage2_step, stage2_rotate),
        #         record_file
        #     )
        #     loss_str += "\n%s " % env_name
        #     for metric, val in score_summary.items():
        #         loss_str += ', %s: %.2f' % (metric, val)
        #         # writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
        # write_to_record_file(
        #     ('\n%s' % loss_str),
        #     record_file
        # )
        # json.dump(
        #     result,
        #     open(os.path.join(args.pred_dir, "eval_detail_%s.json" % env_name), 'w'),
        #     sort_keys=True, indent=4, separators=(',', ': ')
        # )


def main():
    args = parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0 if args.local_rank == -1 else args.local_rank))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cpu")
    args.device = str(device)

    rank = args.rank
    completed = False
    try:
        if args.world_size > 1:
            rank = init_distributed(args)
            args.rank = rank
        if getattr(args, "debug_ddp", False) or is_main_process():
            print(
                f"[DDP] rank={get_rank()} local_rank={args.local_rank} "
                f"world_size={get_world_size()} device={device}"
            )

        set_random_seed(args.seed + rank)

        if args.mode == 'train':
            train_env, val_envs = build_train_dataset(args, rank=rank)
            train(args, train_env, val_envs, rank=rank)
        elif args.mode == 'eval':
            val_envs = build_val_dataset(args, rank=rank)
            distributed_barrier()
            valid(args, val_envs, rank=rank)
            distributed_barrier()
        elif args.mode == 'visualize':
            vis_envs = build_vis_dataset(args, rank=rank)
            distributed_barrier()
            visualize(args, vis_envs, rank=rank)
            distributed_barrier()
        completed = True
    finally:
        cleanup_distributed(synchronize=completed)


if __name__ == '__main__':
    main()
