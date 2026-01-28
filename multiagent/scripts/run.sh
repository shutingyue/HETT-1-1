ngpus=2
seed=0

flag="--root_dir ../datasets

      --world_size ${ngpus}
      --seed ${seed}
      

      --feedback student

      --max_action_len 10
      --max_instr_len 100

      --lr 1e-5
      --iters 200000
      --log_every 2
      --batch_size 12
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5
      
      --nss_w 0.1
      --nss_r 0
      "


NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29533}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# train
# CUDA_VISIBLE_DEVICES='1,2'  python xview_et/main.py --output_dir ../datasets/AVDN/et_v8 $flag\
CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$ngpus --master_port=$PORT \
  main.py --output_dir ../datasets/AVDN/et_v8 $flag\
#       --resume_file ../datasets/AVDN/et_v8/ckpts/latest_dict_18352
# eval
# CUDA_VISIBLE_DEVICES='1'  python xview_et/main.py --output_dir ../datasets/AVDN/et_output $flag \
#       --resume_file ../datasets/AVDN/et_haa/ckpts/best_val_unseen\
#       --inference True
#       --submit True
