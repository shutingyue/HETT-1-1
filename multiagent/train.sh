ngpus=4
seed=0
# export NCCL_DEBUG=INFO

flag="--world_size ${ngpus}
      --seed ${seed}
      --feedback student
      --mode train
      --altitude 50
      --learning_rate 1e-4
      --batch_size 2
      --train_trajectory_type mturk
      --log_every 1
      --eval_every 1
      --epochs 50
      --save_every 1
      --log_dir log
      --move_iteration 10
      --max_action_len 20
      --grid_size 5
      "


NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29533}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# train
# CUDA_VISIBLE_DEVICES='1,2'  python xview_et/main.py --output_dir ../datasets/AVDN/et_v8 $flag\
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.run --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$ngpus --master_port=$PORT \
  main.py $flag

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python   main.py $flag
#       --resume_file ../datasets/AVDN/et_v8/ckpts/latest_dict_18352
# eval
# CUDA_VISIBLE_DEVICES='1'  python xview_et/main.py --output_dir ../datasets/AVDN/et_output $flag \
#       --resume_file ../datasets/AVDN/et_haa/ckpts/best_val_unseen\
#       --inference True
#       --submit True
