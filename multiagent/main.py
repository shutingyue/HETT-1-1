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


def format_region_prompt_logs(logs):
    keys = [
        'region_context_norm',
        'original_emb_norm',
        'region_residual_ratio',
        'region_attn_entropy',
        'region_attn_max',
        'region_prompt_input_tokens',
        'region_prompt_input_grid_size',
        'region_prompt_max_spatial_tokens',
        'region_query_norm',
        'region_query_offdiag_cos',
        'spatial_pos_embed_norm',
        'spatial_pos_embed_offdiag_cos',
        'projected_query_norm',
        'projected_query_offdiag_cos',
        'query_spatial_affinity_target',
        'query_spatial_affinity_mean',
        'query_spatial_affinity_gap',
        'query_spatial_affinity_max',
        'region_attn_diversity_loss',
        'region_gen_attn_entropy',
        'region_gen_attn_max',
        'region_gen_attn_effective_num',
        'region_gen_attn_topk_mass',
        'region_gen_attn_peak_margin',
        'region_gen_attn_offdiag_cos',
        'region_token_offdiag_cos',
        'visual_token_offdiag_cos',
        'raw_visual_token_offdiag_cos',
    ]
    if not any(logs.get(key, []) for key in keys):
        return None
    values = []
    for key in keys:
        logged_values = []
        for value in logs.get(key, []):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                logged_values.append(value)
        if logged_values:
            values.append((key, sum(logged_values) / max(len(logged_values), 1)))
    if not values:
        return None
    count_keys = {
        'region_prompt_input_tokens',
        'region_prompt_input_grid_size',
        'region_prompt_max_spatial_tokens',
    }
    return "region_prompt " + " ".join(
        f"{key} {value:.0f}" if key in count_keys else f"{key} {value:.4f}"
        for key, value in values
    )


def write_region_prompt_logs(args, logs, record_file):
    if not (
        bool(getattr(args, 'use_region_prompt', False))
        and getattr(args, 'region_prompt_mode', 'residual') in ('residual', 'replace')
    ):
        return
    log_line = format_region_prompt_logs(logs)
    if log_line is not None:
        write_to_record_file("\n%s" % log_line, record_file)


def format_stop_visual_context_logs(args, logs):
    if not bool(getattr(args, 'use_stop_visual_context', False)):
        return None

    mode = getattr(args, 'stop_visual_context_mode', 'global_attn')
    if mode == 'global_attn':
        keys = [
            'stop_visual_context_input_tokens',
            'stop_visual_context_input_grid_size',
            'stop_visual_context_norm',
            'global_attn_entropy',
            'global_attn_max',
            'global_attn_effective_num',
            'global_attn_topk_mass',
            'global_attn_peak_margin',
        ]
    elif mode == 'fixed_partition':
        keys = [
            'stop_visual_context_input_tokens',
            'stop_visual_context_input_grid_size',
            'stop_visual_context_norm',
            'fixed_region_token_offdiag_cos',
            'fixed_region_select_entropy',
            'fixed_region_select_max',
            'fixed_region_select_effective_num',
            'fixed_region_select_topk_mass',
            'fixed_region_select_peak_margin',
            'fixed_partition_region_size_min',
            'fixed_partition_region_size_max',
            'fixed_partition_region_size_mean',
        ]
    else:
        return None

    values = []
    for key in keys:
        logged_values = []
        for value in logs.get(key, []):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                logged_values.append(value)
        if logged_values:
            values.append((key, sum(logged_values) / max(len(logged_values), 1)))
    if not values:
        return None
    return "stop_visual_context mode {} ".format(mode) + " ".join(
        f"{key} {value:.0f}" if key in ('stop_visual_context_input_tokens', 'stop_visual_context_input_grid_size')
        else f"{key} {value:.4f}" for key, value in values
    )


def write_stop_visual_context_logs(args, logs, record_file):
    log_line = format_stop_visual_context_logs(args, logs)
    if log_line is not None:
        write_to_record_file("\n%s" % log_line, record_file)


def format_stop_contrast_logs(args, logs):
    if not bool(getattr(args, 'use_stop_contrast', False)):
        return None

    def finite_values(key):
        values = []
        for value in logs.get(key, []):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return values

    def sum_first_available(*keys):
        for key in keys:
            values = finite_values(key)
            if values:
                return sum(values), key
        return 0.0, keys[0]

    def aggregate_skip_reason(num_valid, num_pos, num_neg):
        if num_valid <= 0:
            return 'no_valid'
        if num_pos <= 0:
            return 'no_pos'
        if num_neg <= 0:
            return 'no_neg'
        return 'none'

    local_num_valid, _ = sum_first_available(
        'stop_contrast_local_num_valid',
        'stop_contrast_num_valid',
    )
    local_num_pos, _ = sum_first_available(
        'stop_contrast_local_num_pos',
        'stop_contrast_num_pos',
    )
    local_num_neg, _ = sum_first_available(
        'stop_contrast_local_num_neg',
        'stop_contrast_num_neg',
    )
    global_num_valid, _ = sum_first_available(
        'stop_contrast_global_num_valid',
        'stop_contrast_local_num_valid',
        'stop_contrast_num_valid',
    )
    global_num_pos, _ = sum_first_available(
        'stop_contrast_global_num_pos',
        'stop_contrast_local_num_pos',
        'stop_contrast_num_pos',
    )
    global_num_neg, _ = sum_first_available(
        'stop_contrast_global_num_neg',
        'stop_contrast_local_num_neg',
        'stop_contrast_num_neg',
    )

    values = []
    if local_num_valid > 0 or local_num_pos > 0 or local_num_neg > 0:
        values.extend([
            ('stop_contrast_num_valid', local_num_valid),
            ('stop_contrast_num_pos', local_num_pos),
            ('stop_contrast_num_neg', local_num_neg),
            ('stop_contrast_pos_ratio', local_num_pos / max(local_num_valid, 1.0)),
            ('stop_contrast_local_num_valid', local_num_valid),
            ('stop_contrast_local_num_pos', local_num_pos),
            ('stop_contrast_local_num_neg', local_num_neg),
        ])
    if global_num_valid > 0 or global_num_pos > 0 or global_num_neg > 0:
        values.extend([
            ('stop_contrast_global_num_valid', global_num_valid),
            ('stop_contrast_global_num_pos', global_num_pos),
            ('stop_contrast_global_num_neg', global_num_neg),
            ('stop_contrast_global_pos_ratio', global_num_pos / max(global_num_valid, 1.0)),
        ])

    conversion_count_keys = [
        'stop_contrast_num_strict_pos',
        'stop_contrast_num_hard_neg',
        'stop_contrast_num_easy_neg',
        'stop_contrast_num_ambiguous',
        'stop_contrast_num_ignored',
        'stop_contrast_global_num_hard_neg',
        'stop_contrast_global_num_easy_neg',
        'stop_contrast_global_num_ambiguous',
        'stop_contrast_global_num_ignored',
    ]
    for key in conversion_count_keys:
        count_value, _ = sum_first_available(key)
        if count_value > 0 or finite_values(key):
            values.append((key, count_value))

    scalar_mean_keys = [
        'stop_contrast_loss',
        'stop_contrast_temperature',
        'stop_contrast_lambda',
        'stop_contrast_positive_mode_id',
        'stop_contrast_strict_pos_threshold',
        'stop_contrast_hard_neg_min',
        'stop_contrast_easy_neg_max',
        'stop_contrast_use_easy_negatives',
        'stop_contrast_conversion_pos_ratio',
        'stop_contrast_conversion_neg_ratio',
        'stop_contrast_score_easy_neg_mean',
    ]
    for key in scalar_mean_keys:
        logged_values = finite_values(key)
        if logged_values:
            values.append((key, sum(logged_values) / max(len(logged_values), 1)))

    local_skip_reason = aggregate_skip_reason(local_num_valid, local_num_pos, local_num_neg)
    local_skipped = 1.0 if local_skip_reason != 'none' else 0.0
    skipped_values = finite_values('stop_contrast_local_skipped') or finite_values('stop_contrast_skipped')
    if skipped_values or local_num_valid > 0:
        values.append(('stop_contrast_skipped', local_skipped))
        values.append(('stop_contrast_local_skipped', local_skipped))
    any_rank_values = finite_values('stop_contrast_any_rank_skipped') or skipped_values
    if any_rank_values:
        any_rank_skipped = 1.0 if any(value > 0.0 for value in any_rank_values) else 0.0
        values.append(('stop_contrast_any_rank_skipped', any_rank_skipped))

    pos_mean = 0.0
    neg_mean = 0.0
    pos_weights = finite_values('stop_contrast_local_num_pos') or finite_values('stop_contrast_num_pos')
    neg_weights = finite_values('stop_contrast_local_num_neg') or finite_values('stop_contrast_num_neg')
    pos_values = finite_values('stop_contrast_score_pos_mean')
    neg_values = finite_values('stop_contrast_score_neg_mean')
    if local_num_pos > 0 and pos_values:
        weighted = [
            value * weight
            for value, weight in zip(pos_values, pos_weights)
            if weight > 0
        ]
        pos_mean = sum(weighted) / max(sum(weight for weight in pos_weights if weight > 0), 1.0)
    if local_num_neg > 0 and neg_values:
        weighted = [
            value * weight
            for value, weight in zip(neg_values, neg_weights)
            if weight > 0
        ]
        neg_mean = sum(weighted) / max(sum(weight for weight in neg_weights if weight > 0), 1.0)
    if pos_values:
        values.append(('stop_contrast_score_pos_mean', pos_mean if local_num_pos > 0 else 0.0))
    if neg_values:
        values.append(('stop_contrast_score_neg_mean', neg_mean if local_num_neg > 0 else 0.0))
    if pos_values or neg_values:
        score_gap = pos_mean - neg_mean if local_num_pos > 0 and local_num_neg > 0 else 0.0
        values.append(('stop_contrast_score_gap', score_gap))

    hard_neg_values = finite_values('stop_contrast_score_hard_neg_mean')
    hard_neg_weights = finite_values('stop_contrast_num_hard_neg')
    local_num_hard_neg = sum(hard_neg_weights) if hard_neg_weights else 0.0
    hard_neg_mean = 0.0
    if local_num_hard_neg > 0 and hard_neg_values:
        weighted = [
            value * weight
            for value, weight in zip(hard_neg_values, hard_neg_weights)
            if weight > 0
        ]
        hard_neg_mean = sum(weighted) / max(sum(weight for weight in hard_neg_weights if weight > 0), 1.0)
    if hard_neg_values:
        values.append(('stop_contrast_score_hard_neg_mean', hard_neg_mean if local_num_hard_neg > 0 else 0.0))
        values.append((
            'stop_contrast_score_gap_pos_hard_neg',
            pos_mean - hard_neg_mean if local_num_pos > 0 and local_num_hard_neg > 0 else 0.0,
        ))

    if not values:
        return None
    source = getattr(args, 'stop_contrast_visual_source', 'none')
    local_reason = local_skip_reason
    global_reason = aggregate_skip_reason(global_num_valid, global_num_pos, global_num_neg)
    log_line = "stop_contrast visual_source {} ".format(source) + " ".join(
        f"{key} {value:.0f}" if key in (
            'stop_contrast_num_valid',
            'stop_contrast_num_pos',
            'stop_contrast_num_neg',
            'stop_contrast_skipped',
            'stop_contrast_local_num_valid',
            'stop_contrast_local_num_pos',
            'stop_contrast_local_num_neg',
            'stop_contrast_local_skipped',
            'stop_contrast_any_rank_skipped',
            'stop_contrast_global_num_valid',
            'stop_contrast_global_num_pos',
            'stop_contrast_global_num_neg',
            'stop_contrast_positive_mode_id',
            'stop_contrast_use_easy_negatives',
            'stop_contrast_num_strict_pos',
            'stop_contrast_num_hard_neg',
            'stop_contrast_num_easy_neg',
            'stop_contrast_num_ambiguous',
            'stop_contrast_num_ignored',
            'stop_contrast_global_num_hard_neg',
            'stop_contrast_global_num_easy_neg',
            'stop_contrast_global_num_ambiguous',
            'stop_contrast_global_num_ignored',
        )
        else f"{key} {value:.4f}" for key, value in values
    )
    if local_reason != 'none':
        log_line += f" stop_contrast_skip_reason {local_reason}"
        log_line += f" stop_contrast_local_skip_reason {local_reason}"
    if global_reason != 'none':
        log_line += f" stop_contrast_global_skip_reason {global_reason}"
    return log_line


def write_stop_contrast_logs(args, logs, record_file):
    log_line = format_stop_contrast_logs(args, logs)
    if log_line is not None:
        write_to_record_file("\n%s" % log_line, record_file)


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
        "create={:.2f} update={:.2f} merge={:.2f} create_rate={:.3f} update_rate={:.3f} merge_rate={:.3f} "
        "global_k={:.2f} local_k={:.2f} coverage={:.3f} active_valid={:.3f} empty={:.3f} "
        "goal_rel avg={:.4f} range=[{:.4f},{:.4f}] created_goal={:.4f} updated_goal={:.4f} merged_goal={:.4f} "
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
        with open(os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        # writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

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
            write_region_prompt_logs(args, agent.logs, record_file)
            write_stop_visual_context_logs(args, agent.logs, record_file)
            write_stop_contrast_logs(args, agent.logs, record_file)
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
                    write_region_prompt_logs(args, agent_eval.logs, record_file)
                    write_stop_visual_context_logs(args, agent_eval.logs, record_file)
                    write_stop_contrast_logs(args, agent_eval.logs, record_file)
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
            write_region_prompt_logs(args, agent_eval.logs, record_file)
            write_stop_visual_context_logs(args, agent_eval.logs, record_file)
            write_stop_contrast_logs(args, agent_eval.logs, record_file)
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
