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

        # Train for 2 epochs before evaluate again
        agent.train(loader, args.log_every, feedback=args.feedback,
                    nss_w_weighting=1)  # nss_w_weighting = max(0, (args.iters/2 - idx)/ (args.iters/2)))

        if default_gpu:
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

            # Run validation
            loss_str = "\nepoch {}".format(idx)

            agent.save(idx, os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR, "latest"))
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
