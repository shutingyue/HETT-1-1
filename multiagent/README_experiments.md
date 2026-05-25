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

## 4. ELAM-v1, grid memory only

```bash
NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer --no-use_progress_bar" \
MEMORY_FLAGS="--use_umti --no-enable_topo_memory \
--use_memory_type_embedding \
--use_elam \
--elam_fusion_mode none \
--no-use_time_decay \
--debug_elam" \
SPATIAL_FLAGS="--no-spatial_compression" \
bash train.sh
```

Expected: `use_umti=True`, `use_elam=True`, `enable_topo_memory=False`,
`elam_fusion_mode=none`; ELAM debug prints shapes/losses and train/eval/checkpoint all succeed.

## 5. ELAM-v1 + topo place memory

```bash
NGPUS=2 CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_FLAGS="--batch_size 2 --epochs 1 --max_train_batches_per_epoch 5 --log_every 1 --eval_every 1 --save_every 1 --no-resume_optimizer --no-use_progress_bar" \
MEMORY_FLAGS="--use_umti --enable_topo_memory --persistent_topo_memory \
--no-use_landmark_nodes --no-use_event_nodes \
--use_memory_type_embedding \
--use_topo_gate \
--use_elam \
--elam_fusion_mode none \
--no-use_time_decay \
--debug_elam \
--goal_create_norm_threshold 0.50 \
--retrieve_goal_weight 0.50 --retrieve_visual_weight 0.30 --retrieve_visit_weight 0.20 \
--global_retrieve_k 16" \
SPATIAL_FLAGS="--no-spatial_compression" \
bash train.sh
```

Expected: `use_umti=True`, `enable_topo_memory=True`, `use_elam=True`,
`use_time_decay=False`, `use_landmark_nodes=False`, `use_event_nodes=False`; ELAM sees
grid + topo memory tokens, ELAM loss is finite, and train/eval/checkpoint all succeed.

## UASC-v1 auxiliary controller

UASC-v1 is a lightweight uncertainty-aware stage-and-stop controller. Its first
integration is auxiliary only and does not change the default HETT agent stage
or stop execution rules.

Smoke test:

```bash
python multiagent/test_uasc.py
```

Enable UASC auxiliary training with `--use_uasc --uasc_control_mode aux_only`.
Keep `--no-use_uasc` or omit the flag to preserve the legacy path. The
`stop_only` and `stage_stop` modes are reserved for later inference-control
experiments.
