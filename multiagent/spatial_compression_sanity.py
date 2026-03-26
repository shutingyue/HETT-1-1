import os
import sys
import torch
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiagent.models.ET_haa import ET


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


if __name__ == "__main__":
    run_case(spatial_compression=False)
    run_case(spatial_compression=True)
    print("sanity_check=passed")
