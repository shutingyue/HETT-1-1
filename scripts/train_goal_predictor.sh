CUDA_VISIBLE_DEVICES='0'

cd ..

python main_goal_predictor.py \
    --mode train \
    --model seq2seq_with_map \
    --altitude 50 \
    --learning_rate 0.0015 \
    --train_batch_size 12 \
    --train_trajectory_type mturk \
    --eval_max_timestep 15

python main_goal_predictor.py \
    --mode train \
    --model cma_with_map \
    --altitude 50 \
    --learning_rate 0.0015 \
    --train_batch_size 12 \
    --train_trajectory_type mturk \
    --eval_max_timestep 15 \
    --log