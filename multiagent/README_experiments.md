# UMTI-lite Smoke Tests

Run from `multiagent/`.

## 1. HETT baseline equivalent

```bash
NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer" \
MEMORY_FLAGS="--no-use_umti --no-enable_topo_memory --no-persistent_topo_memory --no-use_time_decay" \
SPATIAL_FLAGS="--no-spatial_compression" \
bash train.sh
```

Expected: `use_umti=False`, `enable_topo_memory=False`, `use_time_decay=False`.

## 2. UMTI-lite only, grid memory only

```bash
NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer" \
MEMORY_FLAGS="--use_umti --no-enable_topo_memory --use_memory_type_embedding --no-use_time_decay --debug_memory_tokens" \
SPATIAL_FLAGS="--no-spatial_compression" \
bash train.sh
```

Expected: `use_umti=True`, `enable_topo_memory=False`; only grid memory tokens are wrapped.

## 3. UMTI-lite + topo place memory, without time decay

```bash
NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer" \
MEMORY_FLAGS="--use_umti --enable_topo_memory --persistent_topo_memory \
--no-use_landmark_nodes --no-use_event_nodes \
--use_memory_type_embedding \
--use_topo_gate \
--no-use_time_decay \
--debug_memory_tokens \
--goal_create_norm_threshold 0.50 \
--retrieve_goal_weight 0.50 --retrieve_visual_weight 0.30 --retrieve_visit_weight 0.20 \
--global_retrieve_k 16" \
SPATIAL_FLAGS="--no-spatial_compression" \
bash train.sh
```

Expected: `use_umti=True`, `enable_topo_memory=True`, `persistent_topo_memory=True`,
`use_memory_type_embedding=True`, `use_topo_gate=True`, `use_time_decay=False`,
`use_landmark_nodes=False`, `use_event_nodes=False`.
