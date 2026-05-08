#!/bin/bash
set -euo pipefail

ngpus=${NGPUS:-1}
seed=0

BASE_FLAGS="--world_size ${ngpus} \
      --seed ${seed} \
      --feedback student \
      --mode train \
      --altitude 50 \
      --learning_rate 1e-4 \
      --batch_size 2 \
      --train_trajectory_type mturk \
      --log_every 1 \
      --eval_every 1 \
      --epochs 25 \
      --save_every 1 \
      --log_dir log \
      --move_iteration 10 \
      --max_action_len 20 \
      --darknet_model_file /mnt/HDD/data/YST/HETT/HETT/datasets/darknet/yolo_v3.cfg \
      --darknet_weight_file /mnt/HDD/data/YST/HETT/HETT/datasets/darknet/yolo_v3.pth \
      --grid_size 5 "

SPATIAL_FLAGS="${SPATIAL_FLAGS:-}"
MEMORY_FLAGS="${MEMORY_FLAGS:-}"

flag="${BASE_FLAGS} ${SPATIAL_FLAGS} ${MEMORY_FLAGS}"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29536}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

echo "Running with flags:"
echo "${flag}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$ngpus \
    --master_port=$PORT \
    main.py $flag

      #  --spatial_compression \
    #  --spatial_dist_threshold 2 \
     # --spatial_far_coarse_size 2"