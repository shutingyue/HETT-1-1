#!/bin/bash
export PYTHONPATH="/mnt/HDD/data/YST/HETT/citynav:/mnt/HDD/data/YST/HETT/HETT:$PYTHONPATH"

# ==========================================
# 增加这一段：封印 GDAL/OpenCV 等底层 C 库的多线程，防止段错误
export GDAL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# ==========================================

ngpus=1
seed=0

# 所有的参数写在这里，注意最后一行千万不要加反斜杠 \
flag="--world_size ${ngpus} \
      --seed ${seed} \
      --feedback student \
      --mode eval \
      --altitude 50 \
      --learning_rate 1e-4 \
      --batch_size 1 \
      --train_trajectory_type mturk \
      --log_every 1 \
      --eval_every 1 \
      --epochs 25 \
      --save_every 1 \
      --log_dir log \
      --move_iteration 10 \
      --max_action_len 20 \
      --grid_size 5 \
      --darknet_model_file /mnt/HDD/data/YST/HETT/HETT/datasets/darknet/yolo_v3.cfg \
      --darknet_weight_file /mnt/HDD/data/YST/HETT/HETT/datasets/darknet/yolo_v3.pth \
      --checkpoint /mnt/HDD/data/YST/HETT/HETT/datasets/checkpoint/best_val_unseen"
      
# 注意上方增加了 --num_workers 0 \ ，强制让 DataLoader 使用单主进程加载数据

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29535}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 指定使用 GPU 0
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$ngpus \
    --master_port=$PORT \
    main.py $flag
