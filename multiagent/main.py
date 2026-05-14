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
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from agent import NavCMTAgent
from env import CityNavBatch
from parser import parse_args

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def distributed_barrier():
    if is_distributed():
        dist.barrier()


EXPERIMENT_CONFIG_GROUPS = [
    ("Experiment", [
        "seed",
        "mode",
        "world_size",
        "local_rank",
        "log_dir",
    ]),
    ("Training", [
        "epochs",
        "batch_size",
        "learning_rate",
        "feedback",
        "eval_every",
        "save_every",
        "log_every",
        "resume_optimizer",
    ]),
    ("Memory", [
        "grid_size",
        "enable_topo_memory",
        "use_topo_memory",
        "persistent_topo_memory",
        "use_time_decay",
        "use_memory_type_embedding",
        "num_memory_types",
        "spatial_compression",
    ]),
    ("Topo Memory", [
        "topo_max_nodes",
        "global_retrieve_k",
        "local_hops",
        "topo_knn",
        "topo_use_graph_encoder",
        "topo_message_passing_layers",
        "topo_merge_radius",
        "topo_create_radius",
        "topo_update_momentum",
    ]),
    ("Topo Retrieval Weights", [
        "retrieve_goal_weight",
        "retrieve_visual_weight",
        "retrieve_visit_weight",
        "goal_create_norm_threshold",
    ]),
    ("Disabled Nodes", [
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


def save_args_json(args, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    args_path = os.path.join(log_dir, "args.json")
    with open(args_path, "w") as outf:
        json.dump(vars(args), outf, indent=4, sort_keys=True, default=str)
    return args_path


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
    update_count = _mean_log_value(logs, 'update_existing_place_nodes_count', _mean_log_value(logs, 'step_updated_place_nodes'))
    merge_count = _mean_log_value(logs, 'merge_place_nodes_count', _mean_log_value(logs, 'step_merged_place_nodes'))
    op_count = max(create_count + update_count, 1e-6)
    create_rate = _mean_log_value(logs, 'create_rate', create_count / op_count)
    update_rate = _mean_log_value(logs, 'update_rate', update_count / op_count)
    merge_rate = _mean_log_value(logs, 'merge_rate', merge_count / op_count)
    global_k = _mean_log_value(logs, 'global_retrieved_nodes')
    retrieval_coverage = _mean_log_value(logs, 'retrieval_coverage', global_k / max(avg_place_nodes, 1.0))
    goal_rel_avg = _mean_log_value(logs, 'goal_rel_raw_avg', _mean_log_value(logs, 'goal_relevance', _mean_log_value(logs, 'avg_goal_relevance')))
    goal_rel_min = _min_log_value(logs, 'goal_rel_raw_min', _min_log_value(logs, 'goal_relevance', goal_rel_avg))
    goal_rel_max = _max_log_value(logs, 'goal_rel_raw_max', _max_log_value(logs, 'goal_relevance', _mean_log_value(logs, 'max_goal_relevance', goal_rel_avg)))
    goal_rel_norm_avg = _mean_log_value(logs, 'goal_rel_norm_avg', _mean_log_value(logs, 'avg_goal_relevance_norm', float('nan')))
    goal_rel_norm_min = _min_log_value(logs, 'goal_rel_norm_min', _min_log_value(logs, 'retrieval_goal_norm_min', goal_rel_norm_avg))
    goal_rel_norm_max = _max_log_value(logs, 'goal_rel_norm_max', _max_log_value(logs, 'max_goal_relevance_norm', goal_rel_norm_avg))
    visual_change_avg = _mean_log_value(logs, 'visual_change')
    visual_change_min = _min_log_value(logs, 'visual_change', visual_change_avg)
    visual_change_max = _max_log_value(logs, 'visual_change', visual_change_avg)
    created_goal = _mean_log_value(
        logs,
        'created_goal_raw',
        _mean_log_value(
            logs,
            'created_goal_relevance',
            _mean_log_value(logs, 'goal_relevance_of_created_nodes', float('nan')),
        ),
    )
    created_goal_norm = _mean_log_value(
        logs,
        'created_goal_norm',
        _mean_log_value(
            logs,
            'created_goal_relevance_norm',
            _mean_log_value(logs, 'goal_relevance_norm_of_created_nodes', float('nan')),
        ),
    )
    updated_goal = _mean_log_value(
        logs,
        'updated_goal_raw',
        _mean_log_value(
            logs,
            'updated_goal_relevance',
            _mean_log_value(logs, 'goal_relevance_of_updated_nodes', float('nan')),
        ),
    )
    updated_goal_norm = _mean_log_value(
        logs,
        'updated_goal_norm',
        _mean_log_value(
            logs,
            'updated_goal_relevance_norm',
            _mean_log_value(logs, 'goal_relevance_norm_of_updated_nodes', float('nan')),
        ),
    )
    merged_goal = _mean_log_value(
        logs,
        'merged_goal_raw',
        _mean_log_value(
            logs,
            'merged_goal_relevance',
            _mean_log_value(logs, 'goal_relevance_of_merged_nodes', float('nan')),
        ),
    )
    merged_goal_norm = _mean_log_value(
        logs,
        'merged_goal_norm',
        _mean_log_value(
            logs,
            'merged_goal_relevance_norm',
            _mean_log_value(logs, 'goal_relevance_norm_of_merged_nodes', float('nan')),
        ),
    )
    topo_line = (
        "[topo_stats] place_nodes avg={:.2f} min={:.2f} max={:.2f} sat={:.3f} "
        "create={:.2f} update={:.2f} merge={:.2f} create_rate={:.3f} update_rate={:.3f} merge_rate={:.3f} "
        "global_k={:.2f} local_k={:.2f} coverage={:.3f} active_valid={:.3f} empty={:.3f} "
        "goal_rel_raw avg={:.4f} range=[{:.4f},{:.4f}] goal_rel_norm avg={:.4f} range=[{:.4f},{:.4f}] "
        "goal_boost_fire={:.3f} created_goal_raw={:.4f} updated_goal_raw={:.4f} merged_goal_raw={:.4f} "
        "created_goal_norm={:.4f} updated_goal_norm={:.4f} merged_goal_norm={:.4f} "
        "base_create_triggers spatial={:.3f} visual={:.3f} turn={:.3f} merge_unrel={:.3f} goal_boost={:.3f} "
        "trigger_goal_norm spatial={:.4f} visual={:.4f} turn={:.4f} merge_unrel={:.4f} goal_boost={:.4f} "
        "retrieval_goal_raw={:.4f} retrieval_goal_norm avg={:.4f} range=[{:.4f},{:.4f}] topk_norm={:.4f} non_topk_norm={:.4f} "
        "retrieval_components all(g/v/visit)={:.4f}/{:.4f}/{:.4f} topk={:.4f}/{:.4f}/{:.4f} non_topk={:.4f}/{:.4f}/{:.4f} "
        "topk_largest goal={:.3f} visual={:.3f} visit={:.3f} "
        "visual_change avg={:.4f} range=[{:.4f},{:.4f}] "
        "token_norm topo={:.4f}+/-{:.4f} global={:.4f} local={:.4f}"
    ).format(
        avg_place_nodes,
        min_place_nodes,
        max_place_nodes,
        _mean_log_value(logs, 'node_saturation_ratio'),
        create_count,
        update_count,
        merge_count,
        create_rate,
        update_rate,
        merge_rate,
        global_k,
        _mean_log_value(logs, 'local_retrieved_nodes'),
        retrieval_coverage,
        _mean_log_value(logs, 'active_node_valid_ratio'),
        _mean_log_value(logs, 'empty_retrieval_ratio'),
        goal_rel_avg,
        goal_rel_min,
        goal_rel_max,
        goal_rel_norm_avg,
        goal_rel_norm_min,
        goal_rel_norm_max,
        _mean_log_value(logs, 'goal_boost_fire_rate', _mean_log_value(logs, 'goal_boost_create')),
        created_goal,
        updated_goal,
        merged_goal,
        created_goal_norm,
        updated_goal_norm,
        merged_goal_norm,
        _mean_log_value(logs, 'spatial_create_rate', _mean_log_value(logs, 'spatial_create')),
        _mean_log_value(logs, 'visual_create_rate', _mean_log_value(logs, 'visual_create')),
        _mean_log_value(logs, 'turn_create_rate', _mean_log_value(logs, 'turn_create')),
        _mean_log_value(logs, 'merge_unreliable_rate', _mean_log_value(logs, 'merge_unreliable')),
        _mean_log_value(logs, 'goal_boost_create_rate', _mean_log_value(logs, 'goal_boost_create')),
        _mean_log_value(logs, 'spatial_create_goal_norm', float('nan')),
        _mean_log_value(logs, 'visual_create_goal_norm', float('nan')),
        _mean_log_value(logs, 'turn_create_goal_norm', float('nan')),
        _mean_log_value(logs, 'merge_unreliable_goal_norm', float('nan')),
        _mean_log_value(logs, 'goal_boost_goal_norm', float('nan')),
        _mean_log_value(logs, 'retrieval_goal_raw_avg', float('nan')),
        _mean_log_value(logs, 'retrieval_goal_norm_avg', float('nan')),
        _min_log_value(logs, 'retrieval_goal_norm_min', float('nan')),
        _max_log_value(logs, 'retrieval_goal_norm_max', float('nan')),
        _mean_log_value(logs, 'retrieval_topk_goal_norm', _mean_log_value(logs, 'topk_goal_norm_mean', float('nan'))),
        _mean_log_value(logs, 'retrieval_non_topk_goal_norm', float('nan')),
        _mean_log_value(logs, 'retrieval_goal_component_avg', float('nan')),
        _mean_log_value(logs, 'retrieval_visual_component_avg', float('nan')),
        _mean_log_value(logs, 'retrieval_visit_component_avg', float('nan')),
        _mean_log_value(logs, 'topk_goal_component_avg', float('nan')),
        _mean_log_value(logs, 'topk_visual_component_avg', float('nan')),
        _mean_log_value(logs, 'topk_visit_component_avg', float('nan')),
        _mean_log_value(logs, 'non_topk_goal_component_avg', float('nan')),
        _mean_log_value(logs, 'non_topk_visual_component_avg', float('nan')),
        _mean_log_value(logs, 'non_topk_visit_component_avg', float('nan')),
        _mean_log_value(logs, 'topk_goal_largest_component_rate', float('nan')),
        _mean_log_value(logs, 'topk_visual_largest_component_rate', float('nan')),
        _mean_log_value(logs, 'topk_visit_largest_component_rate', float('nan')),
        visual_change_avg,
        visual_change_min,
        visual_change_max,
        _mean_log_value(logs, 'topo_token_norm_mean'),
        _mean_log_value(logs, 'topo_token_norm_std'),
        _mean_log_value(logs, 'global_token_norm_mean'),
        _mean_log_value(logs, 'local_token_norm_mean'),
    )
    if logs.get('raw_landmark_count', []) or logs.get('valid_landmark_count', []):
        topo_line += (
            " landmarks raw={:.2f} valid={:.2f} retrieved={:.2f} local={:.2f} "
            "edges={:.2f} degree avg={:.2f} max={:.2f} "
            "conf avg={:.3f} range=[{:.3f},{:.3f}] text_rel={:.3f} geo={:.3f} visual={:.3f} "
            "lm_norm={:.4f} gate={:.3f} ratio={:.3f} empty={:.3f} low_conf={:.2f} all_to_all={:.0f}"
        ).format(
            _mean_log_value(logs, 'raw_landmark_count'),
            _mean_log_value(logs, 'valid_landmark_count'),
            _mean_log_value(logs, 'retrieved_landmark_count'),
            _mean_log_value(logs, 'local_retrieved_landmark_count'),
            _mean_log_value(logs, 'attached_landmark_edges'),
            _mean_log_value(logs, 'avg_landmark_degree'),
            _mean_log_value(logs, 'max_landmark_degree'),
            _mean_log_value(logs, 'landmark_conf_avg'),
            _min_log_value(logs, 'landmark_conf_min'),
            _max_log_value(logs, 'landmark_conf_max'),
            _mean_log_value(logs, 'landmark_text_rel_avg'),
            _mean_log_value(logs, 'landmark_geo_score_avg'),
            _mean_log_value(logs, 'landmark_visual_support_avg'),
            _mean_log_value(logs, 'landmark_token_norm_mean'),
            _mean_log_value(logs, 'landmark_gate_avg'),
            _mean_log_value(logs, 'landmark_place_token_ratio'),
            _mean_log_value(logs, 'landmark_empty_ratio'),
            _mean_log_value(logs, 'landmark_filtered_low_conf_count'),
            _max_log_value(logs, 'landmark_all_to_all_detected'),
        )
        if logs.get('original_landmark_map_norm', []):
            topo_line += " original_lm_map_norm={:.4f}".format(
                _mean_log_value(logs, 'original_landmark_map_norm')
            )
    return topo_line


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
    if rank == 0:
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
    if rank == 0:
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
    if rank == 0:
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
    default_gpu = is_default_gpu(args)

    if default_gpu:
        config_text = print_experiment_config(args)
        save_args_json(args, args.log_dir)
        # writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(config_text + '\n', record_file, verbose=False)

    best_val = {'val_unseen': {"sr": 0., "state": ""}, 'val_unseen_full_traj': {"sr": 0., "state": ""}}

    # first evaluation
    if args.eval_first:
        loss_str = ""
        if default_gpu:

            for env_name, env in val_envs.items():
                agent_class_eval = NavCMTAgent
                agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)

                if args.checkpoint is not None:
                    start_epoch = agent_eval.load(os.path.join(args.checkpoint))
                    if default_gpu:
                        write_to_record_file(
                            "\nLOAD the model from {}, epoch {}".format(args.checkpoint, start_epoch),
                            record_file
                        )

                agent_eval.env = env
                # sampler = DistributedSampler(env, num_replicas=args.world_size, rank=rank)
                loader = DataLoader(env, batch_size=1)
                # Get validation distance from goal under test evaluation conditions
                agent_eval.test(loader, feedback='student')
                pred_results = agent_eval.get_results()

                score_summary, result = env.eval_metrics(pred_results)
                loss_str += ", %s \n" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                if env_name in best_val:
                    if score_summary['sr'] >= best_val[env_name]['sr']:
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Epoch %d %s' % (start_epoch, loss_str)
            write_to_record_file(loss_str, record_file)
        distributed_barrier()

    torch.cuda.empty_cache()
    agent_class = NavCMTAgent
    agent = agent_class(args, rank=rank)

    # resume file
    start_epoch = 0
    if args.checkpoint is not None:
        start_epoch = agent.load(os.path.join(args.checkpoint))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, epoch {}".format(args.checkpoint, start_epoch),
                record_file
            )

    # Start Training
    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start epoch: %s' % str(start_epoch), record_file
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
        agent.train(loader, train_passes, feedback=args.feedback,
                    nss_w_weighting=1,
                    max_batches_per_epoch=max_train_batches)  # nss_w_weighting = max(0, (args.iters/2 - idx)/ (args.iters/2)))

        distributed_barrier()

        if default_gpu:
            should_eval = (idx % max(int(args.eval_every), 1)) == 0
            should_save = (idx % max(int(args.save_every), 1)) == 0
            ml_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)

            direction_loss = sum(agent.logs['direction_loss']) / max(len(agent.logs['direction_loss']), 1)

            progress_loss = sum(agent.logs['progress_loss']) / max(len(agent.logs['progress_loss']), 1)
            goal_predict_loss = sum(agent.logs['goal_predict_loss']) / max(len(agent.logs['goal_predict_loss']), 1)
            # target_predict_loss = sum(agent.logs['target_predict_loss']) / max(len(agent.logs['target_predict_loss']), 1)
            # writer.add_scalar("loss/IL_loss", IL_loss, iter)

            write_to_record_file(
                "\nIL_loss %.4f direction_loss %.4f progress_loss %.4f goal_predict_loss %.4f" % (
                    ml_loss, direction_loss, progress_loss, goal_predict_loss),
                record_file
            )
            stage1_step = sum(agent.logs['stage1_step']) / max(len(agent.logs['stage1_step']), 1)
            stage2_step = sum(agent.logs['stage2_step']) / max(len(agent.logs['stage2_step']), 1)
            stage2_rotate = sum(agent.logs['stage2_rotate']) / max(len(agent.logs['stage2_rotate']), 1)

            write_to_record_file(
                "\nstage %.4f %.4f %.4f" % (
                    stage1_step, stage2_step, stage2_rotate),
                record_file
            )
            write_to_record_file("\n%s" % format_compression_logs(agent.logs), record_file)
            if args.use_topo_memory:
                write_to_record_file("\n%s" % format_topo_logs(agent.logs), record_file)

            if should_save or should_eval:
                agent.save(idx, os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest"))

            # Run validation
            if should_eval:
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
                    agent_eval.test(loader, feedback='student')
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
                    if args.use_topo_memory:
                        write_to_record_file("\n%s" % format_topo_logs(agent_eval.logs), record_file)
                    loss_str += "\n%s " % env_name
                    for metric, val in score_summary.items():
                        loss_str += ', %s: %.2f' % (metric, val)
                        # writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
                    if env_name in best_val:
                        if score_summary['sr'] >= best_val[env_name]['sr']:
                            best_val[env_name]['sr'] = score_summary['sr']
                            best_val[env_name]['state'] = 'Epoch %d %s' % (idx, loss_str)
                            agent_eval.save(idx, os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "best_%s" % (env_name)))

                write_to_record_file(
                    ('\n%s (%d %d%%) %s' % (
                        timeSince(start, float(idx + 1) / args.epochs), idx + 1, float(idx + 1) / args.epochs * 100,
                        loss_str)),
                    record_file
                )
                write_to_record_file("BEST RESULT TILL NOW", record_file)
                for env_name in best_val:
                    write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)
        distributed_barrier()
        torch.cuda.empty_cache()


def valid(args, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)
    if default_gpu:

        agent_class_eval = NavCMTAgent
        agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
        epoch = agent_eval.load(args.checkpoint)
        if args.checkpoint is not None:
            print("Loaded the listener model at epoch %d from %s" % \
                  (epoch, args.checkpoint))
            loss_str = "\nepoch {}".format(epoch)

        with open(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'valid.txt')
        for env_name, env in val_envs.items():
            agent_eval.logs = defaultdict(list)
            agent_eval.env = env
            loader = DataLoader(env, batch_size=1)
            # Get validation distance from goal under test evaluation conditions
            agent_eval.test(loader, feedback='student')
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
            if args.use_topo_memory:
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
    default_gpu = is_default_gpu(args)
    if default_gpu:

        agent_class_eval = NavCMTAgent
        agent_eval = agent_class_eval(args, rank=rank, allow_ngpus=False)
        epoch = agent_eval.load(args.checkpoint)
        if args.checkpoint is not None:
            print("Loaded the listener model at epoch %d from %s" % \
                  (epoch, args.checkpoint))
            loss_str = "\nepoch {}".format(epoch)

        with open(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'valid.txt')
        for env_name, env in vis_envs.items():
            agent_eval.logs = defaultdict(list)
            agent_eval.env = env
            loader = DataLoader(env, batch_size=1)
            # Get validation distance from goal under test evaluation conditions
            agent_eval.visualize(loader, feedback='student')
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
    rank = 0
    # if args.train_val_on_full:
    #     args.max_action_len *= 4
    if args.world_size > 1:
        rank = init_distributed(args)
        # print('success')
        args.local_rank = rank
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0
    # if args.vision_only:
    #     print("!!! Vision only")
    # if args.language_only:
    #     print("!!! Language only")

    set_random_seed(args.seed + rank)

    if args.mode == 'train':
        train_env, val_envs = build_train_dataset(args, rank=rank)
        train(args, train_env, val_envs, rank=rank)
    elif args.mode == 'eval':
        val_envs = build_val_dataset(args, rank=rank)
        valid(args, val_envs, rank=rank)
    elif args.mode == 'visualize':
        vis_envs = build_vis_dataset(args, rank=rank)
        visualize(args, vis_envs, rank=rank)


if __name__ == '__main__':
    main()
