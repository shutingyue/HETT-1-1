"""Utilities for instruction-conditioned topological memory."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def safe_cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute cosine similarity while guarding against zero vectors."""

    a_norm = a / a.norm(dim=dim, keepdim=True).clamp_min(eps)
    b_norm = b / b.norm(dim=dim, keepdim=True).clamp_min(eps)
    return (a_norm * b_norm).sum(dim=dim)


def bearing_change(
    prev_xy: torch.Tensor,
    curr_xy: torch.Tensor,
    next_xy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the turn angle in degrees for a three-point trajectory fragment."""

    vec_a = curr_xy - prev_xy
    vec_b = next_xy - curr_xy
    norm_a = vec_a.norm(dim=-1).clamp_min(eps)
    norm_b = vec_b.norm(dim=-1).clamp_min(eps)
    cos_theta = (vec_a * vec_b).sum(dim=-1) / (norm_a * norm_b)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def score_to_similarity(distance: torch.Tensor, scale: float) -> torch.Tensor:
    """Map a distance-like quantity to a bounded similarity score."""

    return 1.0 / (1.0 + distance / max(scale, 1e-6))


def landmark_name_overlap(lhs: Sequence[str], rhs: Sequence[str]) -> float:
    """Simple Jaccard overlap for two landmark name collections."""

    lhs_set = {name for name in lhs if name}
    rhs_set = {name for name in rhs if name}
    if not lhs_set and not rhs_set:
        return 0.0
    return float(len(lhs_set & rhs_set)) / float(max(len(lhs_set | rhs_set), 1))


def mean_tensor(items: Iterable[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    """Average tensors with a fallback when the iterable is empty."""

    items = list(items)
    if not items:
        return fallback.clone()
    return torch.stack(items, dim=0).mean(dim=0)
