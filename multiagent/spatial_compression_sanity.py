import os
import sys
import torch
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiagent.models.ET_haa import ET
from multiagent.models.region_prompt import RegionPromptAdapter
from multiagent.models.visual_context import StopVisualContextAdapter


def build_args(spatial_compression):
    return SimpleNamespace(
        demb=768,
        encoder_heads=12,
        encoder_layers=2,
        dropout_transformer_encoder=0.1,
        num_input_actions=1,
        dropout_emb=0.0,
        grid_size=5,
        spatial_compression=spatial_compression,
        spatial_dist_threshold=1,
        spatial_far_coarse_size=2,
        enable_topo_memory=False,
        persistent_topo_memory=False,
        use_topo_memory=False,
        use_time_decay=False,
        topo_seed_first_observation=False,
        topo_rebuild_fallback=True,
        use_event_nodes=False,
        use_landmark_nodes=False,
        topo_use_graph_encoder=False,
        topo_aux_grid_supervision=False,
        topo_offset_scale=0.1,
        use_stop_visual_context=False,
        stop_visual_context_mode="global_attn",
        stop_visual_context_dim=768,
        stop_visual_context_num_regions=4,
        stop_visual_context_dropout=0.1,
        stop_visual_context_topk=5,
        use_region_prompt=False,
        region_prompt_mode="residual",
        region_prompt_num=4,
        region_prompt_alpha=0.1,
        region_prompt_dropout=0.1,
        region_prompt_scale_mode="sqrt_dim",
        region_prompt_query_init="pos",
        region_prompt_query_scale=0.1,
        region_prompt_use_pos_embed=True,
        region_prompt_condition_generation=False,
        region_prompt_fuse_instruction=False,
        use_region_attn_diversity=False,
        region_attn_diversity_lambda=0.05,
        region_attn_diversity_mode="cosine_square",
        region_attn_topk=5,
    )


def build_fake_batch(device):
    batch_size = 2
    grid_size = 5
    grid_cell_count = grid_size ** 2
    global_position = torch.stack(
        [torch.tensor([i, j], dtype=torch.float32) for i in range(grid_size) for j in range(grid_size)]
    ) / grid_size

    return {
        "lang": torch.randn(batch_size, 4, 768, device=device),
        "maps": torch.randn(batch_size, 3, 240, 240, device=device),
        # Legacy ET frame-attention path in this sanity case still consumes the
        # original 49-wide frame feature; adapter cases below cover N=25.
        "frames": torch.randn(batch_size, 1, 512, 49, device=device),
        "directions": torch.randn(batch_size, 1, 4, device=device),
        "grid_fts": torch.randn(batch_size, 8, 768, device=device),
        # Same flattened indexing as env.py / agent.py: row * grid_size + col.
        "grid_index": torch.tensor(
            [
                [12, 12, 13, 7, 0, 4, 20, 24],
                [12, 17, 18, 23, 0, 1, 5, 24],
            ],
            dtype=torch.long,
            device=device,
        ),
        "current_grid": torch.tensor([12, 12], dtype=torch.long, device=device),
        "candidates": global_position.unsqueeze(0).repeat(batch_size, 1, 1).to(device),
        "centroids": torch.zeros(batch_size, 0, 2, device=device),
        "lang_cls": torch.randn(batch_size, 49, device=device),
        "time_steps": torch.tensor(
            [
                [0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
                [0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
            ],
            device=device,
        ),
        "current_t": 5,
        "expected_logits_shape": (batch_size, grid_cell_count, 1),
    }


def run_case(spatial_compression):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ET(build_args(spatial_compression=spatial_compression)).to(device)
    model.eval()
    batch = build_fake_batch(device)

    with torch.no_grad():
        _, _, _, target_logits, _, compression_stats = model(**batch)

    print(f"spatial_compression={spatial_compression}")
    print(f"logits_shape={tuple(target_logits.shape)}")
    for key in [
        "tokens_before",
        "tokens_after",
        "near_tokens",
        "far_summary_tokens",
        "merged_away_tokens",
    ]:
        print(f"{key}={compression_stats[key]:.4f}")

    assert tuple(target_logits.shape) == batch["expected_logits_shape"], (
        f"logits shape mismatch: got {tuple(target_logits.shape)}, "
        f"expected {batch['expected_logits_shape']}"
    )

    if spatial_compression:
        assert compression_stats["tokens_after"] < compression_stats["tokens_before"]
        assert compression_stats["near_tokens"] + compression_stats["far_summary_tokens"] == compression_stats["tokens_after"]
        assert compression_stats["merged_away_tokens"] == (
            compression_stats["tokens_before"] - compression_stats["tokens_after"]
        )


def expected_fixed_partition_region_sizes(num_visual_tokens):
    side = int(num_visual_tokens ** 0.5)
    assert side * side == num_visual_tokens, f"N must be square, got {num_visual_tokens}"
    row_mid = side // 2
    col_mid = side // 2
    return [
        row_mid * col_mid,
        row_mid * (side - col_mid),
        (side - row_mid) * col_mid,
        (side - row_mid) * (side - col_mid),
    ]


def expected_corner_indices(num_visual_tokens):
    side = int(num_visual_tokens ** 0.5)
    assert side * side == num_visual_tokens, f"N must be square, got {num_visual_tokens}"
    top_left = 0
    top_right = side - 1
    bottom_left = side * (side - 1)
    bottom_right = side * side - 1
    return [top_left, top_right, bottom_left, bottom_right]


def run_adapter_small_tensor_case(num_visual_tokens, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    time_steps = 1
    visual_dim = 8
    embed_dim = 16
    legacy_lang_dim = 49

    visual_tokens = torch.randn(batch_size, time_steps, visual_dim, num_visual_tokens, device=device)
    lang_cls = torch.randn(batch_size, legacy_lang_dim, device=device)

    stop_adapter = StopVisualContextAdapter(
        mode="fixed_partition",
        visual_dim=visual_dim,
        embed_dim=embed_dim,
        num_heads=4,
        instruction_dim=legacy_lang_dim,
    ).to(device)
    stop_context, stop_diagnostics = stop_adapter(visual_tokens, lang_cls)
    expected_sizes = expected_fixed_partition_region_sizes(num_visual_tokens)

    print(f"adapter_case={label}")
    print(f"visual_tokens_N={num_visual_tokens}")
    print(f"fixed_partition_region_sizes={stop_diagnostics['fixed_partition_region_sizes']}")
    print(f"fixed_partition_region_size_min={stop_diagnostics['fixed_partition_region_size_min']:.2f}")
    print(f"fixed_partition_region_size_max={stop_diagnostics['fixed_partition_region_size_max']:.2f}")
    print(f"fixed_partition_region_size_mean={stop_diagnostics['fixed_partition_region_size_mean']:.2f}")

    assert tuple(stop_context.shape) == (batch_size, time_steps, embed_dim)
    assert stop_diagnostics["fixed_partition_region_sizes"] == expected_sizes
    assert stop_diagnostics["fixed_partition_region_size_min"] == min(expected_sizes)
    assert stop_diagnostics["fixed_partition_region_size_max"] == max(expected_sizes)
    assert stop_diagnostics["fixed_partition_region_size_mean"] == sum(expected_sizes) / len(expected_sizes)

    region_adapter = RegionPromptAdapter(
        visual_dim=visual_dim,
        embed_dim=embed_dim,
        num_region_queries=4,
        num_heads=4,
        instruction_dim=legacy_lang_dim,
        query_init="pos",
        max_spatial_tokens=num_visual_tokens,
    ).to(device)
    region_tokens = region_adapter(visual_tokens[:, 0, :, :], lang_cls)
    expected_indices = expected_corner_indices(num_visual_tokens)
    actual_indices = region_adapter.region_query_pos_indices.cpu().tolist()

    print(f"region_prompt_pos_indices={actual_indices}")
    assert tuple(region_tokens.shape) == (batch_size, 4, embed_dim)
    assert actual_indices == expected_indices


if __name__ == "__main__":
    print("current_hett_experiment_visual_tokens=25 (grid_size=5)")
    run_adapter_small_tensor_case(num_visual_tokens=25, label="main_current_hett")
    print("compatibility_visual_tokens=49 (grid_size=7)")
    run_adapter_small_tensor_case(num_visual_tokens=49, label="compat_legacy_n49")
    run_case(spatial_compression=False)
    run_case(spatial_compression=True)
    print("sanity_check=passed")
