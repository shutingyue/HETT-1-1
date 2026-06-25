import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multiagent.models.stop_contrast import StopContrastLoss


def assert_finite(tensor, name):
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN/inf"


def assert_detached_diagnostics(diagnostics):
    for key, value in diagnostics.items():
        if torch.is_tensor(value):
            assert not value.requires_grad, f"{key} diagnostic should be detached"


def run_source_case(source, instruction_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    time_steps = 4
    hidden_dim = 768
    proj_dim = 64

    module = StopContrastLoss(
        visual_source=source,
        hidden_dim=hidden_dim,
        visual_dim=hidden_dim,
        instruction_dim=49,
        proj_dim=proj_dim,
        temperature=0.07,
        dropout=0.0,
    ).to(device)
    action_hidden = torch.randn(batch_size, time_steps, hidden_dim, device=device, requires_grad=True)
    visual_context = None
    if source != "none":
        visual_context = torch.randn(batch_size, time_steps, hidden_dim, device=device, requires_grad=True)
    instruction = torch.randn(batch_size, instruction_dim, device=device)
    progress = torch.tensor(
        [[0.1, 0.4, 0.82, 0.95], [0.2, 0.9, 0.7, 0.85]],
        dtype=torch.float32,
        device=device,
    )
    valid_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, True]],
        dtype=torch.bool,
        device=device,
    )

    loss, diagnostics, scores = module(
        action_hidden=action_hidden,
        visual_context=visual_context,
        instruction=instruction,
        progress_target=progress,
        valid_mask=valid_mask,
        progress_threshold=0.8,
    )
    assert tuple(scores.shape) == (batch_size, time_steps)
    assert loss.item() > 0.0
    assert_finite(loss, f"{source}/instruction{instruction_dim}/loss")
    assert_finite(scores, f"{source}/instruction{instruction_dim}/scores")
    assert diagnostics["stop_contrast_num_pos"] > 0
    assert diagnostics["stop_contrast_num_neg"] > 0
    assert 0.0 < diagnostics["stop_contrast_pos_ratio"] < 1.0
    assert diagnostics["stop_contrast_skipped"] == 0.0
    assert diagnostics["stop_contrast_skip_reason"] == "none"
    assert diagnostics["stop_contrast_local_num_pos"] == diagnostics["stop_contrast_num_pos"]
    assert diagnostics["stop_contrast_local_num_neg"] == diagnostics["stop_contrast_num_neg"]
    assert diagnostics["stop_contrast_local_skipped"] == 0.0
    assert diagnostics["stop_contrast_local_skip_reason"] == "none"
    assert diagnostics["stop_contrast_any_rank_skipped"] == 0.0
    assert diagnostics["stop_contrast_global_num_valid"] >= diagnostics["stop_contrast_local_num_valid"]
    assert diagnostics["stop_contrast_global_num_pos"] >= diagnostics["stop_contrast_local_num_pos"]
    assert diagnostics["stop_contrast_global_num_neg"] >= diagnostics["stop_contrast_local_num_neg"]
    assert diagnostics["stop_contrast_global_skip_reason"] == "none"
    assert_detached_diagnostics(diagnostics)

    no_pos_progress = torch.zeros_like(progress)
    no_pos_loss, no_pos_diag = module.compute_loss_from_scores(
        scores=scores,
        progress_target=no_pos_progress,
        valid_mask=valid_mask,
        progress_threshold=0.8,
    )
    assert no_pos_loss.item() == 0.0
    assert no_pos_diag["stop_contrast_num_pos"] == 0
    assert no_pos_diag["stop_contrast_num_neg"] == no_pos_diag["stop_contrast_num_valid"]
    assert no_pos_diag["stop_contrast_pos_ratio"] == 0.0
    assert no_pos_diag["stop_contrast_score_pos_mean"] == 0.0
    assert no_pos_diag["stop_contrast_score_gap"] == 0.0
    assert no_pos_diag["stop_contrast_skipped"] == 1.0
    assert no_pos_diag["stop_contrast_skip_reason"] == "no_pos"
    assert no_pos_diag["stop_contrast_local_skipped"] == 1.0
    assert no_pos_diag["stop_contrast_local_skip_reason"] == "no_pos"
    assert no_pos_diag["stop_contrast_any_rank_skipped"] == 1.0
    assert no_pos_diag["stop_contrast_global_num_pos"] >= no_pos_diag["stop_contrast_local_num_pos"]
    assert_finite(no_pos_loss, f"{source}/no_pos_loss")

    no_neg_progress = torch.ones_like(progress)
    no_neg_loss, no_neg_diag = module.compute_loss_from_scores(
        scores=scores,
        progress_target=no_neg_progress,
        valid_mask=valid_mask,
        progress_threshold=0.8,
    )
    assert no_neg_loss.item() == 0.0
    assert no_neg_diag["stop_contrast_num_neg"] == 0
    assert no_neg_diag["stop_contrast_num_pos"] == no_neg_diag["stop_contrast_num_valid"]
    assert no_neg_diag["stop_contrast_pos_ratio"] == 1.0
    assert no_neg_diag["stop_contrast_score_neg_mean"] == 0.0
    assert no_neg_diag["stop_contrast_score_gap"] == 0.0
    assert no_neg_diag["stop_contrast_skipped"] == 1.0
    assert no_neg_diag["stop_contrast_skip_reason"] == "no_neg"
    assert no_neg_diag["stop_contrast_local_skipped"] == 1.0
    assert no_neg_diag["stop_contrast_local_skip_reason"] == "no_neg"
    assert no_neg_diag["stop_contrast_any_rank_skipped"] == 1.0
    assert no_neg_diag["stop_contrast_global_num_neg"] >= no_neg_diag["stop_contrast_local_num_neg"]
    assert_finite(no_neg_loss, f"{source}/no_neg_loss")


def run_detach_visual_case(detach_visual):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = StopContrastLoss(
        visual_source="global_attn",
        hidden_dim=768,
        visual_dim=768,
        instruction_dim=49,
        proj_dim=64,
        temperature=0.07,
        dropout=0.0,
    ).to(device)
    action_hidden = torch.randn(2, 4, 768, device=device, requires_grad=True)
    visual_context = torch.randn(2, 4, 768, device=device, requires_grad=True)
    instruction = torch.randn(2, 768, device=device)
    progress = torch.tensor(
        [[0.1, 0.4, 0.82, 0.95], [0.2, 0.9, 0.7, 0.85]],
        dtype=torch.float32,
        device=device,
    )
    valid_mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    loss, _, _ = module(
        action_hidden=action_hidden,
        visual_context=visual_context,
        instruction=instruction,
        progress_target=progress,
        valid_mask=valid_mask,
        progress_threshold=0.8,
        detach_visual=detach_visual,
    )
    loss.backward()
    assert action_hidden.grad is not None
    visual_grad_norm = (
        0.0 if visual_context.grad is None else float(visual_context.grad.detach().abs().sum().item())
    )
    if detach_visual:
        assert visual_grad_norm == 0.0
    else:
        assert visual_grad_norm > 0.0


if __name__ == "__main__":
    for source in ("none", "global_attn", "fixed_partition", "region_prompt"):
        for instruction_dim in (49, 768):
            run_source_case(source, instruction_dim)
    run_detach_visual_case(detach_visual=False)
    run_detach_visual_case(detach_visual=True)
    print("stop_contrast_sanity=passed")
