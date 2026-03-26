#!/bin/bash
ngpus=2
seed=0
# export NCCL_DEBUG=INFO

# 【修改点】：这里已经帮你把最后那一行的 --output_dir 删掉了
flag="--world_size ${ngpus} \
      --seed ${seed} \
      --feedback student \
      --mode train \
      --altitude 50 \
      --learning_rate 1e-4 \
      --batch_size 2 \
      --train_trajectory_type mturk \
      --log_every 1 \
      --eval_every 1 \
      --epochs 50 \
      --save_every 1 \
      --log_dir log \
      --move_iteration 10 \
      --max_action_len 20 \
      --darknet_model_file /mnt/HDD/data/YST/HETT/HETT/datasets/darknet/yolo_v3.cfg \
      --darknet_weight_file /mnt/HDD/data/YST/HETT/HETT/datasets/darknet/yolo_v3.pth \
      --grid_size 5 \
      --checkpoint ../datasets/checkpoint/latest \
      --resume_optimizer"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29533}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# train
# 【修改】指定两张显卡 0 和 1
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$ngpus \
    --master_port=$PORT \
    main.py $flag
