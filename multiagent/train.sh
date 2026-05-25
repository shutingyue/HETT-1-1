#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

ngpus=${NGPUS:-1}
seed=0

BASE_FLAGS="--world_size ${ngpus} \
      --seed ${seed} \
      --mode train \
      --altitude 50 \
      --log_dir log \
      --move_iteration 10 \
      --max_action_len 20 \
      --darknet_model_file /data1/data/yueshuting/HETT/HETT/datasets/darknet/yolo_v3.cfg \
      --darknet_weight_file /data1/data/yueshuting/HETT/HETT/datasets/darknet/yolo_v3.pth \
      --grid_size 5 "

DEFAULT_TRAIN_FLAGS="--feedback student \
      --learning_rate 1e-4 \
      --batch_size 2 \
      --train_trajectory_type mturk \
      --log_every 1 \
      --no-use_progress_bar \
      --eval_every 1 \
      --epochs 25 \
      --save_every 1 "

TRAIN_FLAGS="${TRAIN_FLAGS:-}"
SPATIAL_FLAGS="${SPATIAL_FLAGS:-}"
MEMORY_FLAGS="${MEMORY_FLAGS:-}"

FLAGS="${BASE_FLAGS} ${DEFAULT_TRAIN_FLAGS} ${TRAIN_FLAGS} ${SPATIAL_FLAGS} ${MEMORY_FLAGS}"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29536}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

echo "Running with flag groups:"
echo "[BASE_FLAGS]"
echo "${BASE_FLAGS}"
echo "[DEFAULT_TRAIN_FLAGS]"
echo "${DEFAULT_TRAIN_FLAGS}"
echo "[TRAIN_FLAGS]"
echo "${TRAIN_FLAGS:-<none>}"
echo "[SPATIAL_FLAGS]"
echo "${SPATIAL_FLAGS:-<none>}"
echo "[MEMORY_FLAGS]"
echo "${MEMORY_FLAGS:-<none>}"

# Smoke tests:
#   Baseline syntax/import smoke:
#     python3 -m py_compile main.py parser.py agent.py models/ET_haa.py models/topo_memory.py models/umti.py
#   HETT baseline equivalent:
#     NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer --no-use_progress_bar" MEMORY_FLAGS="--no-use_umti --no-enable_topo_memory --no-persistent_topo_memory --no-use_time_decay" SPATIAL_FLAGS="--no-spatial_compression" bash train.sh
#   UMTI-lite grid memory only:
#     NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer --no-use_progress_bar" MEMORY_FLAGS="--use_umti --no-enable_topo_memory --use_memory_type_embedding --no-use_time_decay --debug_memory_tokens" SPATIAL_FLAGS="--no-spatial_compression" bash train.sh
#   UMTI-lite + topo place memory, without time decay:
#     NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer --no-use_progress_bar" MEMORY_FLAGS="--use_umti --enable_topo_memory --persistent_topo_memory --no-use_landmark_nodes --no-use_event_nodes --use_memory_type_embedding --use_topo_gate --no-use_time_decay --debug_memory_tokens --goal_create_norm_threshold 0.50 --retrieve_goal_weight 0.50 --retrieve_visual_weight 0.30 --retrieve_visit_weight 0.20 --global_retrieve_k 16" SPATIAL_FLAGS="--no-spatial_compression" bash train.sh
#   ELAM-v1 grid memory only:
#     NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer --no-use_progress_bar" MEMORY_FLAGS="--use_umti --no-enable_topo_memory --use_memory_type_embedding --use_elam --elam_fusion_mode none --no-use_time_decay --debug_elam" SPATIAL_FLAGS="--no-spatial_compression" bash train.sh
#   ELAM-v1 + topo place memory:
#     NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer --no-use_progress_bar" MEMORY_FLAGS="--use_umti --enable_topo_memory --persistent_topo_memory --no-use_landmark_nodes --no-use_event_nodes --use_memory_type_embedding --use_topo_gate --use_elam --elam_fusion_mode none --no-use_time_decay --debug_elam --goal_create_norm_threshold 0.50 --retrieve_goal_weight 0.50 --retrieve_visual_weight 0.30 --retrieve_visit_weight 0.20 --global_retrieve_k 16" SPATIAL_FLAGS="--no-spatial_compression" bash train.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ${PYTHON_BIN:-python} -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$ngpus \
    --master_port=$PORT \
    main.py $FLAGS
