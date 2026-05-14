"""Persistent instruction-conditioned topological memory for HETT."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from .topo_utils import (
    bearing_change,
    landmark_name_overlap,
    mean_tensor,
    safe_cosine_similarity,
    score_to_similarity,
)


EVENT_TYPE_TO_ID = {
    "turn": 0,
    "branch": 1,
    "first_landmark": 2,
    "relevance_jump": 3,
}


@dataclass
class PlaceNode:
    """Sparse place-centric node that persists across an episode."""

    node_id: int
    center_xy: torch.Tensor
    heading_stats: torch.Tensor
    visual_sum: torch.Tensor
    visual_count: float
    recent_patch_bank: List[torch.Tensor] = field(default_factory=list)
    observed_landmarks: List[str] = field(default_factory=list)
    visit_count: float = 0.0
    first_seen_step: int = 0
    last_seen_step: int = 0
    goal_relevance: float = 0.0
    goal_relevance_norm: float = 0.5
    novelty_score: float = 1.0

    def visual_prototype(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return the mean visual representation for this place."""

        visual_sum = self.visual_sum
        if device is not None and dtype is not None:
            visual_sum = visual_sum.to(device=device, dtype=dtype)
        elif device is not None:
            visual_sum = visual_sum.to(device=device)
        elif dtype is not None:
            visual_sum = visual_sum.to(dtype=dtype)
        return visual_sum / max(self.visual_count, 1.0)


@dataclass
class LandmarkNode:
    """Semantic landmark with persistent identity inside an episode."""

    landmark_id: int
    text_tag: str
    polygon: Optional[Sequence[Sequence[float]]]
    semantic_embedding: torch.Tensor
    geometry_stats: torch.Tensor
    attached_place_ids: List[int] = field(default_factory=list)
    attached_place_scores: Dict[int, float] = field(default_factory=dict)
    confidence: float = 0.0
    instruction_relevance: float = 0.0
    geometry_validity: float = 0.0
    supporting_place_score: float = 0.0
    visual_support_score: float = 0.0
    last_seen_step: int = 0
    center_xy: Optional[torch.Tensor] = None


@dataclass
class EventNode:
    """Sparse event marker attached to persistent graph evolution."""

    event_id: int
    event_type: str
    attached_place_id: int
    step_id: int
    score: float


class GraphTokenEncoder(nn.Module):
    """Encode graph nodes into transformer-friendly token vectors."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.place_meta = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.landmark_meta = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.landmark_geo = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.landmark_norm = nn.LayerNorm(hidden_size)
        self.event_meta = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.place_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.landmark_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.event_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.event_type_embed = nn.Embedding(len(EVENT_TYPE_TO_ID), hidden_size)
        self.landmark_learnable_gate = nn.Parameter(torch.tensor(0.2))

    def _resolve_module_spec(
        self,
        module: nn.Module,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.device, torch.dtype]:
        """Resolve the target device/dtype for a specific encoder submodule."""

        param = next(module.parameters())
        if device is None:
            device = param.device
        if dtype is None:
            dtype = param.dtype
        return device, dtype

    def _place_stats_tensor(
        self,
        place: PlaceNode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build place metadata on the encoder device."""

        center_xy = place.center_xy.to(device=device, dtype=dtype)
        return torch.tensor(
            [
                float(center_xy[0].item()),
                float(center_xy[1].item()),
                float(place.goal_relevance_norm),
                float(place.novelty_score),
                float(place.visit_count),
                float(place.first_seen_step),
                float(place.last_seen_step),
                float(len(place.observed_landmarks)),
            ],
            device=device,
            dtype=dtype,
        )

    def _landmark_stats_tensor(
        self,
        landmark: LandmarkNode,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build landmark-aligned tensors on the encoder device."""

        center_xy = landmark.center_xy
        if center_xy is None:
            center_xy = landmark.semantic_embedding.new_zeros(2)
        center_xy = center_xy.to(device=device, dtype=dtype)
        stats = torch.tensor(
            [
                float(center_xy[0].item()),
                float(center_xy[1].item()),
                float(landmark.confidence),
                float(len(landmark.attached_place_ids)),
            ],
            device=device,
            dtype=dtype,
        )
        return center_xy, stats

    def _event_stats_tensor(
        self,
        event: EventNode,
        ref_step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build event metadata on the encoder device."""

        return torch.tensor(
            [
                float(event.attached_place_id),
                float(ref_step - event.step_id),
                float(event.score),
            ],
            device=device,
            dtype=dtype,
        )

    def encode_place(
        self,
        place: PlaceNode,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode a place node token."""

        device, dtype = self._resolve_module_spec(self.place_meta, device=device, dtype=dtype)
        visual = place.visual_prototype(device=device, dtype=dtype)
        stats = self._place_stats_tensor(place, device=device, dtype=dtype)
        place_meta_device = next(self.place_meta.parameters()).device
        assert visual.device == stats.device == place_meta_device, (
            f"encode_place device mismatch: visual={visual.device}, "
            f"stats={stats.device}, place_meta={place_meta_device}"
        )
        return visual + self.place_meta(stats) + self.place_type.to(device=device, dtype=dtype)

    def encode_landmark(
        self,
        landmark: LandmarkNode,
        support_places: Optional[Sequence[PlaceNode]] = None,
        gate_mode: str = "confidence",
        constant_gate: float = 0.2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode a landmark node token."""

        device, dtype = self._resolve_module_spec(self.landmark_meta, device=device, dtype=dtype)
        semantic_embedding = landmark.semantic_embedding.to(device=device, dtype=dtype)
        _, stats = self._landmark_stats_tensor(landmark, device=device, dtype=dtype)
        geometry_stats = landmark.geometry_stats.to(device=device, dtype=dtype)
        if support_places:
            support_embedding = mean_tensor(
                [place.visual_prototype(device=device, dtype=dtype) for place in support_places],
                semantic_embedding,
            )
        else:
            support_embedding = semantic_embedding.new_zeros(semantic_embedding.shape)
        fused = self.landmark_norm(
            semantic_embedding
            + self.landmark_meta(stats)
            + self.landmark_geo(geometry_stats)
            + support_embedding
            + self.landmark_type.to(device=device, dtype=dtype)
        )
        if gate_mode == "learnable":
            gate_value = self.landmark_learnable_gate.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        elif gate_mode == "constant":
            gate_value = semantic_embedding.new_tensor(float(constant_gate)).clamp(0.0, 1.0)
        else:
            gate_value = semantic_embedding.new_tensor(float(landmark.confidence)).clamp(0.0, 1.0)
        return gate_value * fused

    def encode_event(
        self,
        event: EventNode,
        ref_step: int,
        hidden_dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode an event node token."""

        device, hidden_dtype = self._resolve_module_spec(self.event_meta, device=device, dtype=hidden_dtype)
        event_type_id = EVENT_TYPE_TO_ID.get(event.event_type, 0)
        stats = self._event_stats_tensor(event, ref_step, device=device, dtype=hidden_dtype)
        return (
            self.event_type_embed.weight[event_type_id].to(device=device, dtype=hidden_dtype)
            + self.event_meta(stats)
            + self.event_type.to(device=device, dtype=hidden_dtype)
        )


class TopoMemoryGraph:
    """Persistent per-episode graph with incremental online updates."""

    def __init__(self, args, token_encoder: GraphTokenEncoder):
        self.args = args
        self.token_encoder = token_encoder
        self.reset()

    def reset(self) -> None:
        """Clear all graph state for a new episode."""

        self.base_positions: Optional[torch.Tensor] = None
        self.instruction_feat: Optional[torch.Tensor] = None
        self.fallback_feature: Optional[torch.Tensor] = None
        self.place_nodes: List[PlaceNode] = []
        self.landmark_nodes: List[LandmarkNode] = []
        self.event_nodes: List[EventNode] = []
        self.temporal_edges: List[tuple[int, int]] = []
        self.spatial_edges: List[tuple[int, int]] = []
        self.semantic_edges: List[tuple[int, int]] = []
        self.reobservation_edges: List[tuple[int, int]] = []
        self._landmark_index: Dict[str, int] = {}
        self._active_place_id: Optional[int] = None
        self._next_place_id = 0
        self._next_event_id = 0
        self._prev_xy: Optional[torch.Tensor] = None
        self._prev_prev_xy: Optional[torch.Tensor] = None
        self._prev_goal_relevance: Optional[float] = None
        self._step_new_nodes = 0
        self._step_merged_nodes = 0
        self._step_landmark_attachments = 0
        self._raw_landmark_count = 0
        self._filtered_low_conf_count = 0
        self._landmark_text_rel_values: List[float] = []
        self._landmark_geo_score_values: List[float] = []
        self._landmark_visual_support_values: List[float] = []
        self._last_retrieved_landmark_count = 0
        self._last_retrieved_landmark_norm_mean = 0.0
        self._last_retrieved_landmark_gate_avg = 0.0
        self._step_updated_nodes = 0
        self._step_created_goal_relevance: List[float] = []
        self._step_updated_goal_relevance: List[float] = []
        self._step_merged_goal_relevance: List[float] = []
        self._step_created_goal_relevance_norm: List[float] = []
        self._step_updated_goal_relevance_norm: List[float] = []
        self._step_merged_goal_relevance_norm: List[float] = []
        self._last_update_stats: Dict[str, float] = {}

    def start_episode(
        self,
        base_positions: torch.Tensor,
        instruction_feat: torch.Tensor,
        fallback_feature: torch.Tensor,
    ) -> None:
        """Initialize persistent graph metadata for a new episode."""

        self.reset()
        self.base_positions = base_positions.detach().clone()
        self.instruction_feat = instruction_feat.detach().clone()
        self.fallback_feature = fallback_feature.detach().clone()

    def _require_started(self) -> None:
        """Ensure the episode graph has been initialized."""

        if self.base_positions is None or self.instruction_feat is None or self.fallback_feature is None:
            raise RuntimeError("TopoMemoryGraph.start_episode(...) must be called before update/retrieve.")

    def _place_slot_by_id(self, place_id: int) -> int:
        """Map a persistent place id to its current storage slot."""

        for slot, place in enumerate(self.place_nodes):
            if place.node_id == place_id:
                return slot
        raise KeyError(f"Unknown place node id: {place_id}")

    def _place_by_id(self, place_id: int) -> PlaceNode:
        """Fetch a place node by persistent id."""

        return self.place_nodes[self._place_slot_by_id(place_id)]

    def _resolve_device(
        self,
        ref_tensor: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.device:
        """Resolve the canonical device for topo export/retrieval."""

        if device is not None:
            return device
        if ref_tensor is not None:
            return ref_tensor.device
        return next(self.token_encoder.parameters()).device

    def _resolve_dtype(
        self,
        ref_tensor: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.dtype:
        """Resolve the floating dtype for topo export/retrieval."""

        if dtype is not None:
            return dtype
        if ref_tensor is not None and ref_tensor.is_floating_point():
            return ref_tensor.dtype
        return next(self.token_encoder.parameters()).dtype

    def _to_device(
        self,
        tensor: torch.Tensor,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Move a tensor to the requested device, preserving non-floating dtypes."""

        if dtype is not None and tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    def _mean_tensor_on_device(
        self,
        items: Sequence[torch.Tensor],
        fallback: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Average tensors after aligning them to the export device."""

        moved_items = [self._to_device(item, device, dtype) for item in items]
        return mean_tensor(moved_items, self._to_device(fallback, device, dtype))

    def _normalize_feature_matrix(
        self,
        x: torch.Tensor,
        name: str = "node_features",
    ) -> torch.Tensor:
        """Normalize exported feature matrices to canonical [N, D] shape."""

        if x.dim() == 1:
            return x.unsqueeze(0)
        if x.dim() == 2:
            return x
        if x.dim() == 3 and x.shape[0] == 1:
            return x.squeeze(0)
        if x.dim() == 3 and x.shape[1] == 1:
            return x.squeeze(1)
        raise ValueError(
            f"{name} must have shape [D], [N, D], [N, 1, D], or [1, N, D]; got {tuple(x.shape)}"
        )

    def _normalize_token_2d(self, token: torch.Tensor, name: str = "token") -> torch.Tensor:
        """Backward-compatible alias for canonical [N, D] token normalization."""

        return self._normalize_feature_matrix(token, name=name)

    def _clamp01_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """Clamp a scalar tensor into a score-like [0, 1] range."""

        return value.clamp(0.0, 1.0)

    def _goal_relevance_score(self, goal_relevance: torch.Tensor) -> torch.Tensor:
        """Map raw cosine relevance to a sharper [0, 1] score."""

        temperature = max(float(getattr(self.args, "goal_relevance_temperature", 0.10)), 1e-6)
        return torch.sigmoid(goal_relevance / temperature).clamp(0.0, 1.0)

    def _normalized_novelty(self, novelty: torch.Tensor) -> torch.Tensor:
        """Use bounded visual novelty in create/retrieval bookkeeping."""

        return self._clamp01_tensor(novelty)

    def _active_visual_change(self, feature_t: torch.Tensor) -> torch.Tensor:
        """Measure current observation change relative to the active place node."""

        if not self.place_nodes or self._active_place_id is None:
            return feature_t.new_tensor(1.0)
        active_place = self._place_by_id(self._active_place_id)
        visual_change = 1.0 - safe_cosine_similarity(
            feature_t.unsqueeze(0),
            active_place.visual_prototype().unsqueeze(0),
        ).squeeze(0)
        return self._clamp01_tensor(visual_change)

    def _weighted_unit_score(self, terms: Sequence[tuple[float, torch.Tensor]]) -> torch.Tensor:
        """Combine bounded score terms while tolerating zero/negative user weights."""

        total_weight = sum(max(float(weight), 0.0) for weight, _ in terms)
        if total_weight <= 1e-6:
            return terms[0][1].new_tensor(0.0)
        score = terms[0][1].new_tensor(0.0)
        for weight, value in terms:
            score = score + max(float(weight), 0.0) * value
        return score / total_weight

    def _create_score(
        self,
        novelty: torch.Tensor,
        goal_relevance_score: torch.Tensor,
        visual_change: torch.Tensor,
    ) -> torch.Tensor:
        """Goal-aware create score for turning observations into meaningful places."""

        # TODO: add a continuous turn_signal term here once heading confidence is stable
        # enough to be used as a score instead of a boolean event trigger.
        return self._weighted_unit_score(
            (
                (getattr(self.args, "create_novelty_weight", 0.40), self._normalized_novelty(novelty)),
                (getattr(self.args, "create_goal_weight", 0.40), self._clamp01_tensor(goal_relevance_score)),
                (getattr(self.args, "create_visual_weight", 0.20), self._clamp01_tensor(visual_change)),
            )
        )

    def _mean_float(self, values: Sequence[float]) -> float:
        """Mean helper for sparse debug stats."""

        return float(sum(values) / max(len(values), 1)) if values else 0.0

    def _mean_float_or_nan(self, values: Sequence[float]) -> float:
        """Mean helper for event-conditioned debug stats."""

        return float(sum(values) / len(values)) if values else float("nan")

    def _token_norm_stats(self, tokens: torch.Tensor) -> Dict[str, float]:
        """Return norm mean/std for a [N, D] token matrix."""

        if tokens.numel() == 0 or tokens.shape[0] == 0:
            return {"mean": 0.0, "std": 0.0}
        norms = tokens.norm(dim=-1)
        return {
            "mean": float(norms.mean().item()),
            "std": float(norms.std(unbiased=False).item()) if norms.numel() > 1 else 0.0,
        }

    def _proposal_flags(
        self,
        xy_t: torch.Tensor,
        feature_t: torch.Tensor,
        goal_relevance: torch.Tensor,
        novelty: torch.Tensor,
        landmark_names: Sequence[str],
        heading_change_deg: float,
    ) -> Dict[str, bool]:
        """Evaluate event-triggered new-node proposals."""

        if not self.place_nodes or self._active_place_id is None:
            return {"cold_start": True}

        latest_place = self._place_by_id(self._active_place_id)
        spatial_distance = torch.norm(xy_t - latest_place.center_xy).item()
        visual_distance = 1.0 - safe_cosine_similarity(
            feature_t.unsqueeze(0),
            latest_place.visual_prototype().unsqueeze(0),
        ).item()
        relevance_jump = abs(float(goal_relevance.item()) - float(latest_place.goal_relevance))
        new_landmark = any(name and name not in self._landmark_index for name in landmark_names)
        nearest_existing = min(torch.norm(xy_t - place.center_xy).item() for place in self.place_nodes)
        return {
            "visual_shift": visual_distance >= float(self.args.new_node_vis_threshold),
            "turn_event": heading_change_deg >= float(self.args.turn_event_threshold_deg),
            "relevance_jump": relevance_jump >= float(self.args.relevance_jump_threshold),
            "new_landmark": new_landmark,
            "merge_radius": nearest_existing > float(self.args.place_merge_radius),
            "novelty": float(novelty.item()) >= float(self.args.topo_novelty_threshold),
            "spatial_shift": spatial_distance >= float(self.args.topo_create_radius),
        }

    def _association_scores(
        self,
        xy_t: torch.Tensor,
        feature_t: torch.Tensor,
        goal_relevance: torch.Tensor,
        landmark_names: Sequence[str],
    ) -> List[tuple[int, torch.Tensor]]:
        """Compute multi-cue association scores for existing nodes."""

        scores: List[tuple[int, torch.Tensor]] = []
        for place in self.place_nodes:
            geo_sim = score_to_similarity(
                torch.norm(xy_t - place.center_xy),
                float(self.args.place_merge_radius),
            )
            vis_sim = 0.5 * (
                safe_cosine_similarity(
                    feature_t.unsqueeze(0),
                    place.visual_prototype().unsqueeze(0),
                ).squeeze(0)
                + 1.0
            )
            goal_consistency = feature_t.new_tensor(
                max(0.0, 1.0 - abs(float(goal_relevance.item()) - float(place.goal_relevance)))
            )
            landmark_sim = feature_t.new_tensor(
                landmark_name_overlap(landmark_names, place.observed_landmarks)
            )
            assoc = (
                float(self.args.geo_weight) * geo_sim
                + float(self.args.vis_weight) * vis_sim
                + float(self.args.goal_weight) * goal_consistency
                + float(self.args.sem_weight) * landmark_sim
            )
            scores.append((place.node_id, assoc))
        return scores

    def _create_place_node(
        self,
        xy_t: torch.Tensor,
        heading_vec: torch.Tensor,
        feature_t: torch.Tensor,
        step_id: int,
        goal_relevance: torch.Tensor,
        goal_relevance_score: torch.Tensor,
        novelty: torch.Tensor,
        landmark_names: Sequence[str],
    ) -> int:
        """Create a new place node with a persistent node id."""

        place_id = self._next_place_id
        self._next_place_id += 1
        self.place_nodes.append(
            PlaceNode(
                node_id=place_id,
                center_xy=xy_t.clone(),
                heading_stats=heading_vec.clone(),
                visual_sum=feature_t.clone(),
                visual_count=1.0,
                recent_patch_bank=[feature_t.clone()],
                observed_landmarks=[name for name in landmark_names if name],
                visit_count=1.0,
                first_seen_step=step_id,
                last_seen_step=step_id,
                goal_relevance=float(goal_relevance.item()),
                goal_relevance_norm=float(goal_relevance_score.item()),
                novelty_score=float(novelty.item()),
            )
        )
        self._step_new_nodes += 1
        self._step_created_goal_relevance.append(float(goal_relevance.item()))
        self._step_created_goal_relevance_norm.append(float(goal_relevance_score.item()))
        return place_id

    def _update_place_node(
        self,
        place_id: int,
        xy_t: torch.Tensor,
        heading_vec: torch.Tensor,
        feature_t: torch.Tensor,
        step_id: int,
        goal_relevance: torch.Tensor,
        goal_relevance_score: torch.Tensor,
        novelty: torch.Tensor,
        landmark_names: Sequence[str],
    ) -> int:
        """Merge the current observation into an existing place node."""

        place = self._place_by_id(place_id)
        momentum = float(self.args.topo_update_momentum)
        center_momentum = max(momentum, 0.85)
        place.center_xy = center_momentum * place.center_xy + (1.0 - center_momentum) * xy_t
        place.heading_stats = momentum * place.heading_stats + (1.0 - momentum) * heading_vec
        place.visual_sum = place.visual_sum + feature_t
        place.visual_count += 1.0
        place.visit_count += 1.0
        previous_last_seen = place.last_seen_step
        place.last_seen_step = step_id
        place.goal_relevance = momentum * place.goal_relevance + (1.0 - momentum) * float(goal_relevance.item())
        place.goal_relevance_norm = (
            momentum * place.goal_relevance_norm
            + (1.0 - momentum) * float(goal_relevance_score.item())
        )
        place.novelty_score = momentum * place.novelty_score + (1.0 - momentum) * float(novelty.item())
        for name in landmark_names:
            if name and name not in place.observed_landmarks:
                place.observed_landmarks.append(name)
        place.recent_patch_bank.append(feature_t.clone())
        if len(place.recent_patch_bank) > int(self.args.patch_bank_size):
            place.recent_patch_bank.pop(0)
        self._step_updated_nodes += 1
        self._step_updated_goal_relevance.append(float(goal_relevance.item()))
        self._step_updated_goal_relevance_norm.append(float(goal_relevance_score.item()))
        if step_id - previous_last_seen > 1 and self._active_place_id is not None:
            self.reobservation_edges.append((self._active_place_id, place_id))
        return place_id

    def _create_event(self, event_type: str, place_id: int, step_id: int, score: float) -> None:
        """Attach a persistent event node to the graph."""

        if not bool(self.args.use_event_nodes):
            return
        if len(self.event_nodes) >= int(self.args.max_event_nodes):
            return
        self.event_nodes.append(
            EventNode(
                event_id=self._next_event_id,
                event_type=event_type,
                attached_place_id=place_id,
                step_id=step_id,
                score=float(score),
            )
        )
        self._next_event_id += 1

    def _landmarks_enabled(self) -> bool:
        """Return whether topo landmark nodes should be built or retrieved."""

        return bool(getattr(self.args, "use_landmark_nodes", False)) and (
            getattr(self.args, "topo_landmark_fusion_mode", "aux") != "off"
        )

    def _landmark_gate_value(self, landmark: LandmarkNode) -> float:
        """Mirror GraphTokenEncoder gate selection for debug stats."""

        gate_mode = getattr(self.args, "landmark_gate_mode", "confidence")
        if gate_mode == "constant":
            return max(0.0, min(1.0, float(getattr(self.args, "landmark_constant_gate", 0.2))))
        if gate_mode == "learnable":
            return max(0.0, min(1.0, float(self.token_encoder.landmark_learnable_gate.detach().item())))
        return max(0.0, min(1.0, float(landmark.confidence)))

    def _landmark_polygon_stats(
        self,
        polygon: Optional[Sequence[Sequence[float]]],
        center_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute compact geometry stats and validity for a landmark polygon."""

        device = center_xy.device
        dtype = center_xy.dtype
        zero_stats = torch.tensor(
            [float(center_xy[0].item()), float(center_xy[1].item()), 0.0, 0.0, 0.0, 0.0],
            device=device,
            dtype=dtype,
        )
        center_valid = bool(torch.isfinite(center_xy).all().item()) and bool(
            ((center_xy >= 0.0) & (center_xy <= 1.0)).all().item()
        )
        if not center_valid or polygon is None:
            return zero_stats, 0.0, 0.0
        try:
            points = torch.tensor(polygon, device=device, dtype=dtype)
        except (TypeError, ValueError):
            return zero_stats, 0.0, 0.0
        if points.dim() != 2 or points.shape[0] < 3 or points.shape[1] < 2 or not torch.isfinite(points).all():
            return zero_stats, 0.0, 0.0
        points = points[:, :2]
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * torch.abs(torch.dot(x, torch.roll(y, shifts=-1)) - torch.dot(y, torch.roll(x, shifts=-1)))
        bbox_min = points.min(dim=0).values
        bbox_max = points.max(dim=0).values
        bbox = (bbox_max - bbox_min).clamp_min(0.0)
        map_scale = max(float(getattr(self.args, "map_meters", 1.0)), 1.0)
        area_norm = float((area / (map_scale * map_scale)).clamp_min(0.0).item())
        bbox_w_norm = float((bbox[0] / map_scale).clamp_min(0.0).item())
        bbox_h_norm = float((bbox[1] / map_scale).clamp_min(0.0).item())
        # CityNav landmark contours are in map/world units while centers are normalized.
        # Keep the geometry validity deliberately loose, then rely on confidence and top-k
        # attachment to avoid large polygons becoming dense planning nodes.
        geometry_validity = 1.0 if (area_norm > 1e-8 and area_norm <= 0.75) else 0.0
        geometry_score = geometry_validity * max(0.0, min(1.0, 1.0 / (1.0 + math.sqrt(max(area_norm, 0.0)))))
        stats = torch.tensor(
            [
                float(center_xy[0].item()),
                float(center_xy[1].item()),
                max(0.0, min(1.0, area_norm)),
                max(0.0, min(1.0, bbox_w_norm)),
                max(0.0, min(1.0, bbox_h_norm)),
                geometry_validity,
            ],
            device=device,
            dtype=dtype,
        )
        return stats, float(geometry_validity), float(geometry_score)

    def _landmark_support_scores(self, place_id: int) -> tuple[float, float]:
        """Estimate place and visual support using projected place/instruction relevance."""

        place = self._place_by_id(place_id)
        place_feature = place.visual_prototype(device=self.instruction_feat.device, dtype=self.instruction_feat.dtype)
        place_relevance = safe_cosine_similarity(
            place_feature.unsqueeze(0),
            self.instruction_feat.unsqueeze(0),
        ).squeeze(0)
        support = float((0.5 * (place_relevance + 1.0)).clamp(0.0, 1.0).item())
        if not bool(getattr(self.args, "landmark_use_visual_support", True)):
            return support, 0.0
        return support, support

    def _landmark_confidence(
        self,
        instruction_relevance: float,
        geometry_validity: float,
        supporting_place_score: float,
        visual_support_score: float,
    ) -> float:
        """Confidence filter for semantic landmarks before retrieval."""

        terms = (
            (0.4, instruction_relevance),
            (0.2, geometry_validity),
            (0.2, supporting_place_score),
            (0.2, visual_support_score),
        )
        return max(0.0, min(1.0, sum(weight * max(0.0, min(1.0, value)) for weight, value in terms)))

    def _landmark_attach_score(self, landmark: LandmarkNode, place: PlaceNode) -> tuple[float, float, float]:
        """Score a sparse landmark-place attachment with geometry, instruction, and visual cues."""

        if landmark.center_xy is None:
            geo_score = 0.0
        else:
            center_xy = landmark.center_xy.to(device=place.center_xy.device, dtype=place.center_xy.dtype)
            distance = torch.norm(place.center_xy - center_xy)
            geo_score = float(score_to_similarity(distance, float(self.args.landmark_attach_radius)).item())
            if float(distance.item()) <= float(self.args.landmark_attach_radius):
                geo_score = max(geo_score, 0.75)
        visual_score = 0.0
        if bool(getattr(self.args, "landmark_use_visual_support", True)):
            place_feature = place.visual_prototype(device=self.instruction_feat.device, dtype=self.instruction_feat.dtype)
            rel = safe_cosine_similarity(
                place_feature.unsqueeze(0),
                self.instruction_feat.unsqueeze(0),
            ).squeeze(0)
            visual_score = float((0.5 * (rel + 1.0)).clamp(0.0, 1.0).item())
        score = 0.45 * geo_score + 0.35 * float(landmark.instruction_relevance) + 0.20 * visual_score
        return float(score), float(geo_score), float(visual_score)

    def _refresh_landmark_attachments(self, landmark: LandmarkNode) -> None:
        """Attach each landmark to a sparse, explainable top-k set of place nodes."""

        if not self.place_nodes:
            landmark.attached_place_ids = []
            landmark.attached_place_scores = {}
            return
        max_degree = max(int(getattr(self.args, "max_landmark_degree", 3)), 1)
        scored_places = []
        for place in self.place_nodes:
            attach_score, geo_score, visual_score = self._landmark_attach_score(landmark, place)
            scored_places.append((place.node_id, attach_score, geo_score, visual_score))
        scored_places.sort(key=lambda item: item[1], reverse=True)
        kept = scored_places[:max_degree]
        old_ids = set(landmark.attached_place_ids)
        landmark.attached_place_ids = [place_id for place_id, _, _, _ in kept]
        landmark.attached_place_scores = {place_id: score for place_id, score, _, _ in kept}
        self._step_landmark_attachments += len(set(landmark.attached_place_ids) - old_ids)
        self._landmark_geo_score_values.extend([geo_score for _, _, geo_score, _ in kept])
        self._landmark_visual_support_values.extend([visual_score for _, _, _, visual_score in kept])

    def _enforce_place_landmark_degree(self) -> None:
        """Limit how many landmark anchors any one place keeps."""

        if not self.landmark_nodes:
            return
        place_limit = max(int(getattr(self.args, "landmark_retrieve_k", 2)), 1)
        place_to_landmarks: Dict[int, List[tuple[int, float]]] = {}
        for landmark in self.landmark_nodes:
            for place_id in landmark.attached_place_ids:
                score = float(landmark.attached_place_scores.get(place_id, landmark.confidence))
                place_to_landmarks.setdefault(place_id, []).append((landmark.landmark_id, score))
        allowed = set()
        for place_id, values in place_to_landmarks.items():
            values.sort(key=lambda item: item[1], reverse=True)
            for landmark_id, _ in values[:place_limit]:
                allowed.add((landmark_id, place_id))
        for landmark in self.landmark_nodes:
            kept_ids = [place_id for place_id in landmark.attached_place_ids if (landmark.landmark_id, place_id) in allowed]
            landmark.attached_place_ids = kept_ids
            landmark.attached_place_scores = {
                place_id: score
                for place_id, score in landmark.attached_place_scores.items()
                if place_id in kept_ids
            }
        self.semantic_edges = []
        seen_edges = set()
        for landmark in self.landmark_nodes:
            for place_id in landmark.attached_place_ids:
                edge = (place_id, landmark.landmark_id)
                if edge not in seen_edges:
                    self.semantic_edges.append(edge)
                    seen_edges.add(edge)

    def _update_landmarks(self, place_id: int, step_id: int, landmark_info: Sequence[dict]) -> None:
        """Update persistent landmark nodes and semantic attachments."""

        if not self._landmarks_enabled():
            return
        self._raw_landmark_count += len(landmark_info)
        for item in landmark_info:
            name = item.get("name", "")
            if not name:
                continue
            centroid = item.get("centroid")
            polygon = item.get("polygon")
            if centroid is None:
                self._filtered_low_conf_count += 1
                continue
            elif isinstance(centroid, torch.Tensor):
                centroid_tensor = centroid.to(device=self.instruction_feat.device, dtype=self.instruction_feat.dtype)
            else:
                centroid_tensor = torch.tensor(
                    centroid,
                    device=self.instruction_feat.device,
                    dtype=self.instruction_feat.dtype,
                )
            if centroid_tensor.numel() < 2 or not torch.isfinite(centroid_tensor).all():
                self._filtered_low_conf_count += 1
                continue
            centroid_tensor = centroid_tensor.flatten()[:2]
            geometry_stats, geometry_validity, geometry_score = self._landmark_polygon_stats(polygon, centroid_tensor)
            if geometry_validity <= 0.0:
                self._filtered_low_conf_count += 1
                continue
            instruction_relevance = float(item.get("instruction_relevance", 1.0))
            supporting_place_score, visual_support_score = self._landmark_support_scores(place_id)
            confidence = self._landmark_confidence(
                instruction_relevance,
                geometry_validity,
                supporting_place_score,
                visual_support_score,
            )
            self._landmark_text_rel_values.append(instruction_relevance)
            self._landmark_geo_score_values.append(geometry_score)
            self._landmark_visual_support_values.append(visual_support_score)
            if confidence < float(getattr(self.args, "landmark_conf_threshold", 0.35)):
                self._filtered_low_conf_count += 1
                continue

            if name in self._landmark_index:
                landmark = self.landmark_nodes[self._landmark_index[name]]
                landmark.last_seen_step = step_id
                landmark.confidence = max(float(landmark.confidence), confidence)
                landmark.center_xy = centroid_tensor
                landmark.polygon = polygon
                landmark.geometry_stats = geometry_stats
                landmark.instruction_relevance = max(float(landmark.instruction_relevance), instruction_relevance)
                landmark.geometry_validity = geometry_validity
                landmark.supporting_place_score = max(float(landmark.supporting_place_score), supporting_place_score)
                landmark.visual_support_score = max(float(landmark.visual_support_score), visual_support_score)
            else:
                if len(self.landmark_nodes) >= int(self.args.max_landmark_nodes):
                    continue
                landmark_id = len(self.landmark_nodes)
                self.landmark_nodes.append(
                    LandmarkNode(
                        landmark_id=landmark_id,
                        text_tag=name,
                        polygon=polygon,
                        semantic_embedding=self.instruction_feat.clone(),
                        geometry_stats=geometry_stats,
                        attached_place_ids=[],
                        attached_place_scores={},
                        confidence=confidence,
                        instruction_relevance=instruction_relevance,
                        geometry_validity=geometry_validity,
                        supporting_place_score=supporting_place_score,
                        visual_support_score=visual_support_score,
                        last_seen_step=step_id,
                        center_xy=centroid_tensor,
                    )
                )
                self._landmark_index[name] = landmark_id
                self._create_event("first_landmark", place_id, step_id, 0.5)
            self._refresh_landmark_attachments(self.landmark_nodes[self._landmark_index[name]])
        self._enforce_place_landmark_degree()

    def _remap_edges(self, remove_id: int, keep_id: int) -> None:
        """Redirect persistent edges after a node merge."""

        def remap(edge_list: List[tuple[int, int]]) -> List[tuple[int, int]]:
            remapped: List[tuple[int, int]] = []
            seen = set()
            for src, dst in edge_list:
                src = keep_id if src == remove_id else src
                dst = keep_id if dst == remove_id else dst
                if src == dst:
                    continue
                edge = (src, dst)
                if edge not in seen:
                    remapped.append(edge)
                    seen.add(edge)
            return remapped

        self.temporal_edges = remap(self.temporal_edges)
        self.spatial_edges = remap(self.spatial_edges)
        self.reobservation_edges = remap(self.reobservation_edges)

    def _enforce_max_place_nodes(self) -> None:
        """Keep graph size bounded while preserving persistent ids."""

        max_nodes = int(self.args.max_place_nodes)
        if len(self.place_nodes) <= max_nodes:
            return

        while len(self.place_nodes) > max_nodes:
            if len(self.place_nodes) <= 1:
                return
            raw_features = torch.stack([node.visual_prototype() for node in self.place_nodes], dim=0)
            raw_feature_shape = tuple(raw_features.shape)
            features = self._normalize_feature_matrix(
                raw_features,
                name="_enforce_max_place_nodes.features",
            )
            centers = torch.stack([node.center_xy for node in self.place_nodes], dim=0)
            distance = torch.cdist(centers, centers)
            distance.fill_diagonal_(float("inf"))
            similarity = safe_cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
            if similarity.dim() != 2 or similarity.shape[0] != similarity.shape[1]:
                raise ValueError(
                    "_enforce_max_place_nodes expected square similarity matrix, "
                    f"got similarity shape {tuple(similarity.shape)} from raw feature shape "
                    f"{raw_feature_shape} normalized to {tuple(features.shape)}"
                )
            similarity.fill_diagonal_(0.0)
            redundancy = similarity / (1.0 + distance)
            max_visit = max(float(node.visit_count) for node in self.place_nodes)
            importance = torch.tensor(
                [
                    0.50 * max(0.0, min(1.0, float(node.goal_relevance_norm)))
                    + 0.30 * (float(node.visit_count) / max(max_visit, 1.0))
                    + 0.15 * max(0.0, min(1.0, float(node.novelty_score)))
                    + (0.05 if node.node_id == self._active_place_id else 0.0)
                    for node in self.place_nodes
                ],
                device=features.device,
                dtype=features.dtype,
            )
            remove_slot = int(torch.argmin(importance).item())
            keep_slot = int(torch.argmax(redundancy[remove_slot]).item())
            if keep_slot == remove_slot:
                keep_slot = int(torch.argmin(distance[remove_slot]).item())

            keep_node = self.place_nodes[keep_slot]
            remove_node = self.place_nodes[remove_slot]
            keep_node.center_xy = 0.5 * (keep_node.center_xy + remove_node.center_xy)
            keep_node.visual_sum = keep_node.visual_sum + remove_node.visual_sum
            keep_node.visual_count += remove_node.visual_count
            keep_node.visit_count += remove_node.visit_count
            keep_node.goal_relevance = max(keep_node.goal_relevance, remove_node.goal_relevance)
            keep_node.goal_relevance_norm = max(
                keep_node.goal_relevance_norm,
                remove_node.goal_relevance_norm,
            )
            keep_node.novelty_score = max(keep_node.novelty_score, remove_node.novelty_score)
            keep_node.observed_landmarks = sorted(
                set(keep_node.observed_landmarks + remove_node.observed_landmarks)
            )
            keep_node.recent_patch_bank.extend(remove_node.recent_patch_bank)
            keep_node.recent_patch_bank = keep_node.recent_patch_bank[-int(self.args.patch_bank_size):]
            keep_node.first_seen_step = min(keep_node.first_seen_step, remove_node.first_seen_step)
            keep_node.last_seen_step = max(keep_node.last_seen_step, remove_node.last_seen_step)

            remove_id = remove_node.node_id
            keep_id = keep_node.node_id
            for landmark in self.landmark_nodes:
                updated_ids: List[int] = []
                updated_scores: Dict[int, float] = {}
                for place_id in landmark.attached_place_ids:
                    mapped = keep_id if place_id == remove_id else place_id
                    if mapped not in updated_ids:
                        updated_ids.append(mapped)
                    score = float(landmark.attached_place_scores.get(place_id, landmark.confidence))
                    updated_scores[mapped] = max(score, float(updated_scores.get(mapped, 0.0)))
                landmark.attached_place_ids = updated_ids
                landmark.attached_place_scores = updated_scores
            for event in self.event_nodes:
                if event.attached_place_id == remove_id:
                    event.attached_place_id = keep_id
            self._remap_edges(remove_id, keep_id)
            if self._active_place_id == remove_id:
                self._active_place_id = keep_id
            del self.place_nodes[remove_slot]
            self._step_merged_nodes += 1
            self._step_merged_goal_relevance.append(float(remove_node.goal_relevance))
            self._step_merged_goal_relevance_norm.append(float(remove_node.goal_relevance_norm))

    def _refresh_spatial_edges(self) -> None:
        """Refresh sparse spatial edges after an incremental update."""

        self.spatial_edges = []
        if len(self.place_nodes) <= 1:
            return
        centers = torch.stack([place.center_xy for place in self.place_nodes], dim=0)
        dist = torch.cdist(centers, centers)
        for src_slot, src in enumerate(self.place_nodes):
            for dst_slot in range(src_slot + 1, len(self.place_nodes)):
                if float(dist[src_slot, dst_slot].item()) <= float(self.args.place_merge_radius) * 2.0:
                    self.spatial_edges.append((src.node_id, self.place_nodes[dst_slot].node_id))

    def update_step(
        self,
        observation_feature: torch.Tensor,
        cell_id: int,
        xy: Optional[torch.Tensor],
        landmark_info: Sequence[dict],
        step_id: int,
        time_decay_rate: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Incrementally update the persistent graph with one new observation."""

        self._require_started()
        self._step_new_nodes = 0
        self._step_merged_nodes = 0
        self._step_landmark_attachments = 0
        self._step_updated_nodes = 0
        self._step_created_goal_relevance = []
        self._step_updated_goal_relevance = []
        self._step_merged_goal_relevance = []
        self._step_created_goal_relevance_norm = []
        self._step_updated_goal_relevance_norm = []
        self._step_merged_goal_relevance_norm = []

        base_xy = self.base_positions[int(cell_id)]
        if xy is not None and torch.isfinite(xy).all():
            xy_t = xy.to(device=self.base_positions.device, dtype=self.base_positions.dtype)
        else:
            xy_t = base_xy

        feature_t = observation_feature.to(self.instruction_feat.dtype)
        # Time decay is fully disabled when `time_decay_rate` is None, so the graph update
        # uses the raw observation feature with no learned temporal scaling side effect.
        if time_decay_rate is not None:
            decay = torch.exp(-time_decay_rate * feature_t.new_tensor(0.0))
            feature_t = feature_t * decay
        goal_relevance = safe_cosine_similarity(
            feature_t.unsqueeze(0),
            self.instruction_feat.unsqueeze(0),
        ).squeeze(0)

        if self.place_nodes:
            place_visuals = torch.stack([place.visual_prototype() for place in self.place_nodes], dim=0)
            novelty = 1.0 - safe_cosine_similarity(feature_t.unsqueeze(0), place_visuals).max()
        else:
            novelty = feature_t.new_tensor(1.0)
        novelty_score = self._normalized_novelty(novelty)
        goal_relevance_score = self._goal_relevance_score(goal_relevance)
        visual_change = self._active_visual_change(feature_t)
        create_score = self._create_score(novelty, goal_relevance_score, visual_change)
        goal_relevance_value = float(goal_relevance.item())
        goal_relevance_score_value = float(goal_relevance_score.item())
        novelty_score_value = float(novelty_score.item())
        visual_change_value = float(visual_change.item())
        create_score_value = float(create_score.item())

        heading_vec = feature_t.new_zeros(2)
        heading_change_deg = 0.0
        if self._prev_xy is not None:
            delta = xy_t - self._prev_xy
            heading_vec = delta / delta.norm().clamp_min(1e-6)
        if self._prev_prev_xy is not None and self._prev_xy is not None:
            heading_change_deg = float(
                bearing_change(
                    self._prev_prev_xy.unsqueeze(0),
                    self._prev_xy.unsqueeze(0),
                    xy_t.unsqueeze(0),
                )[0].item()
            )

        landmark_names = [item.get("name", "") for item in landmark_info]
        proposal_flags = self._proposal_flags(
            xy_t=xy_t,
            feature_t=feature_t,
            goal_relevance=goal_relevance,
            novelty=novelty,
            landmark_names=landmark_names,
            heading_change_deg=heading_change_deg,
        )
        association_scores = self._association_scores(
            xy_t=xy_t,
            feature_t=feature_t,
            goal_relevance=goal_relevance,
            landmark_names=landmark_names,
        )

        best_assoc = 0.0
        base_create = not association_scores
        goal_boost_create = False
        spatial_create = False
        visual_create = False
        turn_create = False
        merge_unreliable = False
        event_debug_create = False
        create_new = False

        if not association_scores:
            create_new = True
            selected_place_id = self._create_place_node(
                xy_t,
                heading_vec,
                feature_t,
                step_id,
                goal_relevance,
                goal_relevance_score,
                novelty,
                landmark_names,
            )
        else:
            best_place_id, best_score_tensor = max(association_scores, key=lambda item: float(item[1].item()))
            best_assoc = float(best_score_tensor.item())
            spatial_create = (
                bool(proposal_flags.get("spatial_shift", False))
                or bool(proposal_flags.get("merge_radius", False))
            )
            visual_create = (
                bool(proposal_flags.get("visual_shift", False))
                or bool(proposal_flags.get("novelty", False))
            )
            turn_create = bool(proposal_flags.get("turn_event", False))
            merge_unreliable = best_assoc < float(self.args.topo_merge_sim_threshold)
            base_create = spatial_create or visual_create or turn_create or merge_unreliable

            goal_create_norm_threshold = float(getattr(self.args, "goal_create_norm_threshold", 0.55))
            visual_change_low_threshold = float(
                getattr(self.args, "goal_visual_change_low_threshold", 0.18)
            )
            goal_boost_create = (
                goal_relevance_score_value > goal_create_norm_threshold
                and visual_change_value > visual_change_low_threshold
            )
            create_new = base_create or goal_boost_create

            # Landmark/relevance jump events can still be recorded below, but they no longer
            # suppress normal place-only growth or replace the base geometry/visual create path.
            event_debug_create = (
                bool(proposal_flags.get("turn_event", False))
                or bool(proposal_flags.get("relevance_jump", False))
            )
            if create_new:
                selected_place_id = self._create_place_node(
                    xy_t,
                    heading_vec,
                    feature_t,
                    step_id,
                    goal_relevance,
                    goal_relevance_score,
                    novelty,
                    landmark_names,
                )
            else:
                selected_place_id = self._update_place_node(
                    best_place_id,
                    xy_t,
                    heading_vec,
                    feature_t,
                    step_id,
                    goal_relevance,
                    goal_relevance_score,
                    novelty,
                    landmark_names,
                )

        if self._active_place_id is not None and self._active_place_id != selected_place_id:
            self.temporal_edges.append((self._active_place_id, selected_place_id))
            self._create_event("branch", selected_place_id, step_id, goal_relevance_value)
        self._active_place_id = selected_place_id

        if heading_change_deg >= float(self.args.turn_event_threshold_deg):
            self._create_event("turn", selected_place_id, step_id, heading_change_deg / 180.0)
        if self._prev_goal_relevance is not None:
            jump = abs(goal_relevance_value - self._prev_goal_relevance)
            if jump >= float(self.args.relevance_jump_threshold):
                self._create_event("relevance_jump", selected_place_id, step_id, jump)
        self._prev_goal_relevance = goal_relevance_value

        self._update_landmarks(selected_place_id, step_id, landmark_info)
        self._enforce_max_place_nodes()
        self._refresh_spatial_edges()
        self._prev_prev_xy = self._prev_xy
        self._prev_xy = xy_t

        update_denominator = max(float(self._step_new_nodes + self._step_updated_nodes), 1.0)
        spatial_create_goal_norm = goal_relevance_score_value if spatial_create else float("nan")
        visual_create_goal_norm = goal_relevance_score_value if visual_create else float("nan")
        turn_create_goal_norm = goal_relevance_score_value if turn_create else float("nan")
        merge_unreliable_goal_norm = goal_relevance_score_value if merge_unreliable else float("nan")
        goal_boost_goal_norm = goal_relevance_score_value if goal_boost_create else float("nan")
        self._last_update_stats = {
            "active_place_id": float(selected_place_id),
            "step_new_place_nodes": float(self._step_new_nodes),
            "step_merged_place_nodes": float(self._step_merged_nodes),
            "step_landmark_attachments": float(self._step_landmark_attachments),
            "step_updated_place_nodes": float(self._step_updated_nodes),
            "total_place_nodes": float(len(self.place_nodes)),
            "total_landmark_nodes": float(len(self.landmark_nodes)),
            "total_event_nodes": float(len(self.event_nodes)),
            "total_temporal_edges": float(len(self.temporal_edges)),
            "total_spatial_edges": float(len(self.spatial_edges)),
            "total_semantic_edges": float(len(self.semantic_edges)),
            "total_reobservation_edges": float(len(self.reobservation_edges)),
            "create_place_nodes_count": float(self._step_new_nodes),
            "merge_place_nodes_count": float(self._step_merged_nodes),
            "update_existing_place_nodes_count": float(self._step_updated_nodes),
            "goal_relevance": goal_relevance_value,
            "goal_relevance_score": goal_relevance_score_value,
            "goal_rel_raw_avg": goal_relevance_value,
            "goal_rel_raw_min": goal_relevance_value,
            "goal_rel_raw_max": goal_relevance_value,
            "goal_rel_norm_avg": goal_relevance_score_value,
            "goal_rel_norm_min": goal_relevance_score_value,
            "goal_rel_norm_max": goal_relevance_score_value,
            "created_goal_relevance": self._mean_float_or_nan(self._step_created_goal_relevance),
            "updated_goal_relevance": self._mean_float_or_nan(self._step_updated_goal_relevance),
            "merged_goal_relevance": self._mean_float_or_nan(self._step_merged_goal_relevance),
            "created_goal_relevance_norm": self._mean_float_or_nan(self._step_created_goal_relevance_norm),
            "updated_goal_relevance_norm": self._mean_float_or_nan(self._step_updated_goal_relevance_norm),
            "merged_goal_relevance_norm": self._mean_float_or_nan(self._step_merged_goal_relevance_norm),
            "created_goal_raw": self._mean_float_or_nan(self._step_created_goal_relevance),
            "updated_goal_raw": self._mean_float_or_nan(self._step_updated_goal_relevance),
            "merged_goal_raw": self._mean_float_or_nan(self._step_merged_goal_relevance),
            "created_goal_norm": self._mean_float_or_nan(self._step_created_goal_relevance_norm),
            "updated_goal_norm": self._mean_float_or_nan(self._step_updated_goal_relevance_norm),
            "merged_goal_norm": self._mean_float_or_nan(self._step_merged_goal_relevance_norm),
            "goal_relevance_of_created_nodes": self._mean_float_or_nan(self._step_created_goal_relevance),
            "goal_relevance_of_updated_nodes": self._mean_float_or_nan(self._step_updated_goal_relevance),
            "goal_relevance_of_merged_nodes": self._mean_float_or_nan(self._step_merged_goal_relevance),
            "goal_relevance_norm_of_created_nodes": self._mean_float_or_nan(self._step_created_goal_relevance_norm),
            "goal_relevance_norm_of_updated_nodes": self._mean_float_or_nan(self._step_updated_goal_relevance_norm),
            "goal_relevance_norm_of_merged_nodes": self._mean_float_or_nan(self._step_merged_goal_relevance_norm),
            "novelty_score": novelty_score_value,
            "visual_change": visual_change_value,
            "best_association_score": float(best_assoc),
            "base_create": float(base_create),
            "goal_boost_create": float(goal_boost_create),
            "goal_boost_fire_rate": float(goal_boost_create),
            "final_create": float(create_new),
            "spatial_create": float(spatial_create),
            "visual_create": float(visual_create),
            "turn_create": float(turn_create),
            "merge_unreliable": float(merge_unreliable),
            "spatial_create_rate": float(spatial_create),
            "visual_create_rate": float(visual_create),
            "turn_create_rate": float(turn_create),
            "merge_unreliable_rate": float(merge_unreliable),
            "goal_boost_create_rate": float(goal_boost_create),
            "spatial_create_goal_norm": spatial_create_goal_norm,
            "visual_create_goal_norm": visual_create_goal_norm,
            "turn_create_goal_norm": turn_create_goal_norm,
            "merge_unreliable_goal_norm": merge_unreliable_goal_norm,
            "goal_boost_goal_norm": goal_boost_goal_norm,
            "event_debug_create": float(event_debug_create),
            "create_score": create_score_value,
            "create_rate": float(self._step_new_nodes) / update_denominator,
            "update_rate": float(self._step_updated_nodes) / update_denominator,
            "merge_rate": float(self._step_merged_nodes) / update_denominator,
        }
        return self._last_update_stats

    def build_from_history(
        self,
        history_features: torch.Tensor,
        history_cell_ids: torch.Tensor,
        history_times: torch.Tensor,
        history_xy: torch.Tensor,
        landmark_history: Optional[Sequence[Sequence[dict]]],
        current_grid: int,
        time_decay_rate: Optional[torch.Tensor],
    ) -> None:
        """Fallback debug path that rebuilds the graph by replaying history."""

        self._require_started()
        self.place_nodes = []
        self.landmark_nodes = []
        self.event_nodes = []
        self.temporal_edges = []
        self.spatial_edges = []
        self.semantic_edges = []
        self.reobservation_edges = []
        self._landmark_index = {}
        self._active_place_id = None
        self._next_place_id = 0
        self._next_event_id = 0
        self._prev_xy = None
        self._prev_prev_xy = None
        self._prev_goal_relevance = None

        for step_idx in range(history_features.shape[0]):
            step_xy = None
            if history_xy.numel() > 0 and history_xy.shape[0] > step_idx:
                step_xy = history_xy[step_idx]
            step_landmarks = landmark_history[step_idx] if landmark_history is not None and step_idx < len(landmark_history) else []
            step_cell_id = int(history_cell_ids[step_idx].item()) if history_cell_ids.numel() > step_idx else int(current_grid)
            self.update_step(
                observation_feature=history_features[step_idx],
                cell_id=step_cell_id,
                xy=step_xy,
                landmark_info=step_landmarks,
                step_id=int(history_times[step_idx].item()) if history_times.numel() > step_idx else step_idx,
                time_decay_rate=time_decay_rate,
            )

        if not self.place_nodes:
            self.update_step(
                observation_feature=self.fallback_feature,
                cell_id=int(current_grid),
                xy=self.base_positions[int(current_grid)],
                landmark_info=[],
                step_id=0,
                time_decay_rate=time_decay_rate,
            )

    def _select_landmarks_for_places(
        self,
        place_ids: Sequence[int],
        limit: Optional[int] = None,
        active_only: bool = False,
    ) -> List[LandmarkNode]:
        """Retrieve landmark anchors attached to already-selected place nodes."""

        if not self._landmarks_enabled() or not self.landmark_nodes:
            return []
        if limit is None:
            limit = int(getattr(self.args, "landmark_retrieve_k", 2))
        if limit <= 0:
            return []
        selected_place_ids = set(place_ids)
        if active_only and place_ids:
            selected_place_ids = {int(place_ids[0])}
        threshold = float(getattr(self.args, "landmark_conf_threshold", 0.35))
        candidates = []
        for landmark in self.landmark_nodes:
            attached = selected_place_ids.intersection(landmark.attached_place_ids)
            if not attached or float(landmark.confidence) < threshold:
                continue
            attach_score = max(
                float(landmark.attached_place_scores.get(place_id, 0.0))
                for place_id in attached
            )
            score = 0.65 * float(landmark.confidence) + 0.35 * attach_score
            candidates.append((score, landmark))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [landmark for _, landmark in candidates[:limit]]

    def _encode_landmark_tokens(
        self,
        landmarks: Sequence[LandmarkNode],
        device: torch.device,
        dtype: torch.dtype,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """Encode selected landmark anchors with attached place support."""

        tokens = []
        for landmark in landmarks:
            support_places = [
                self._place_by_id(place_id)
                for place_id in landmark.attached_place_ids
                if any(place.node_id == place_id for place in self.place_nodes)
            ]
            tokens.append(
                self.token_encoder.encode_landmark(
                    landmark,
                    support_places=support_places,
                    gate_mode=getattr(self.args, "landmark_gate_mode", "confidence"),
                    constant_gate=float(getattr(self.args, "landmark_constant_gate", 0.2)),
                    device=device,
                    dtype=dtype,
                )
            )
        if not tokens:
            return fallback.new_zeros((0, fallback.shape[-1]))
        return torch.stack(tokens, dim=0)

    def retrieve_global_graph_tokens(
        self,
        instruction_feat: torch.Tensor,
        k: int,
        current_feature: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, object]:
        """Retrieve top-k global graph tokens for coarse prediction."""

        self._require_started()
        device = self._resolve_device(ref_tensor=instruction_feat, device=device)
        hidden_dtype = self._resolve_dtype(instruction_feat, dtype=dtype)
        self.token_encoder.to(device=device, dtype=hidden_dtype)
        instruction_feat = self._to_device(instruction_feat, device, hidden_dtype)
        if not self.place_nodes:
            empty_tokens = self._normalize_feature_matrix(
                instruction_feat.new_zeros((0, instruction_feat.shape[-1])),
                name="retrieve_global_graph_tokens.place_tokens[empty]",
            )
            empty_positions = self._to_device(self.base_positions[:0].clone(), device, self.base_positions.dtype)
            place_ids = torch.zeros(0, device=device, dtype=torch.long)
            return {
                "place_tokens": empty_tokens,
                "place_positions": empty_positions,
                "place_ids": place_ids,
                "landmark_tokens": self._normalize_feature_matrix(
                    empty_tokens.new_zeros((0, empty_tokens.shape[-1])),
                    name="retrieve_global_graph_tokens.landmark_tokens[empty]",
                ),
                "event_tokens": self._normalize_feature_matrix(
                    empty_tokens.new_zeros((0, empty_tokens.shape[-1])),
                    name="retrieve_global_graph_tokens.event_tokens[empty]",
                ),
                "retrieval_scores": empty_tokens.new_zeros((0,)),
                "retrieval_goal_raw_avg": float("nan"),
                "retrieval_goal_norm_avg": float("nan"),
                "retrieval_goal_norm_min": float("nan"),
                "retrieval_goal_norm_max": float("nan"),
                "topk_goal_norm_mean": float("nan"),
                "retrieval_topk_goal_norm": float("nan"),
                "retrieval_non_topk_goal_norm": float("nan"),
                "retrieval_goal_component_avg": float("nan"),
                "retrieval_visual_component_avg": float("nan"),
                "retrieval_visit_component_avg": float("nan"),
                "topk_goal_component_avg": float("nan"),
                "topk_visual_component_avg": float("nan"),
                "topk_visit_component_avg": float("nan"),
                "non_topk_goal_component_avg": float("nan"),
                "non_topk_visual_component_avg": float("nan"),
                "non_topk_visit_component_avg": float("nan"),
                "topk_goal_largest_component_rate": float("nan"),
                "topk_visual_largest_component_rate": float("nan"),
                "topk_visit_largest_component_rate": float("nan"),
            }

        ranked_ids = []
        if current_feature is None:
            current_feature = self.fallback_feature
        current_feature = self._to_device(current_feature, device, hidden_dtype)
        max_visit = max(float(place.visit_count) for place in self.place_nodes)
        retrieve_goal_weight = float(getattr(self.args, "retrieve_goal_weight", 0.50))
        retrieve_visual_weight = float(getattr(self.args, "retrieve_visual_weight", 0.30))
        retrieve_visit_weight = float(getattr(self.args, "retrieve_visit_weight", 0.20))
        for place in self.place_nodes:
            goal_raw = float(place.goal_relevance)
            goal_score = max(0.0, min(1.0, float(place.goal_relevance_norm)))
            visual_score = 0.5 * (
                safe_cosine_similarity(
                    place.visual_prototype(device=device, dtype=hidden_dtype).unsqueeze(0),
                    current_feature.unsqueeze(0),
                ).item()
                + 1.0
            )
            visit_score = float(place.visit_count) / max(max_visit, 1.0)
            goal_component = retrieve_goal_weight * goal_score
            visual_component = retrieve_visual_weight * visual_score
            visit_component = retrieve_visit_weight * visit_score
            score = goal_component + visual_component + visit_component
            ranked_ids.append(
                (
                    place.node_id,
                    score,
                    goal_raw,
                    goal_score,
                    goal_component,
                    visual_component,
                    visit_component,
                )
            )
        ranked_ids.sort(key=lambda item: item[1], reverse=True)
        top_k = min(max(int(k), 1), len(ranked_ids))
        selected_ranked = ranked_ids[:top_k]
        selected_ids = [node_id for node_id, _, _, _, _, _, _ in selected_ranked]
        selected_places = [self._place_by_id(node_id) for node_id in selected_ids]
        retrieval_scores = torch.tensor(
            [score for _, score, _, _, _, _, _ in selected_ranked],
            device=device,
            dtype=hidden_dtype,
        )
        all_goal_raw = [goal_raw for _, _, goal_raw, _, _, _, _ in ranked_ids]
        all_goal_norm = [goal_norm for _, _, _, goal_norm, _, _, _ in ranked_ids]
        all_goal_components = [goal_component for _, _, _, _, goal_component, _, _ in ranked_ids]
        all_visual_components = [visual_component for _, _, _, _, _, visual_component, _ in ranked_ids]
        all_visit_components = [visit_component for _, _, _, _, _, _, visit_component in ranked_ids]
        selected_goal_norm = [goal_norm for _, _, _, goal_norm, _, _, _ in selected_ranked]
        selected_goal_components = [goal_component for _, _, _, _, goal_component, _, _ in selected_ranked]
        selected_visual_components = [visual_component for _, _, _, _, _, visual_component, _ in selected_ranked]
        selected_visit_components = [visit_component for _, _, _, _, _, _, visit_component in selected_ranked]
        non_topk_ranked = ranked_ids[top_k:]
        non_topk_goal_norm = [goal_norm for _, _, _, goal_norm, _, _, _ in non_topk_ranked]
        non_topk_goal_components = [goal_component for _, _, _, _, goal_component, _, _ in non_topk_ranked]
        non_topk_visual_components = [visual_component for _, _, _, _, _, visual_component, _ in non_topk_ranked]
        non_topk_visit_components = [visit_component for _, _, _, _, _, _, visit_component in non_topk_ranked]
        selected_largest_components = []
        for _, _, _, _, goal_component, visual_component, visit_component in selected_ranked:
            if goal_component >= visual_component and goal_component >= visit_component:
                selected_largest_components.append("goal")
            elif visual_component >= visit_component:
                selected_largest_components.append("visual")
            else:
                selected_largest_components.append("visit")
        selected_largest_denominator = max(float(len(selected_largest_components)), 1.0)

        place_tokens = self._normalize_feature_matrix(
            torch.stack(
                [self.token_encoder.encode_place(place, device=device, dtype=hidden_dtype) for place in selected_places],
                dim=0,
            ),
            name="retrieve_global_graph_tokens.place_tokens",
        )
        place_positions = torch.stack(
            [self._to_device(place.center_xy, device, self.base_positions.dtype) for place in selected_places],
            dim=0,
        )
        place_ids = torch.tensor(selected_ids, device=device, dtype=torch.long)
        selected_set = set(selected_ids)

        selected_landmarks = self._select_landmarks_for_places(
            selected_ids,
            limit=int(getattr(self.args, "landmark_retrieve_k", 2)),
            active_only=False,
        )
        landmark_tokens_tensor = self._normalize_feature_matrix(
            self._encode_landmark_tokens(selected_landmarks, device, hidden_dtype, place_tokens),
            name="retrieve_global_graph_tokens.landmark_tokens",
        )
        self._last_retrieved_landmark_count = int(landmark_tokens_tensor.shape[0])
        self._last_retrieved_landmark_norm_mean = self._token_norm_stats(landmark_tokens_tensor)["mean"]
        self._last_retrieved_landmark_gate_avg = self._mean_float(
            [self._landmark_gate_value(landmark) for landmark in selected_landmarks]
        )
        ref_step = max(place.last_seen_step for place in self.place_nodes)
        event_tokens = [
            self.token_encoder.encode_event(event, ref_step, hidden_dtype, device)
            for event in self.event_nodes
            if event.attached_place_id in selected_set
        ]
        return {
            "place_tokens": place_tokens,
            "place_positions": place_positions,
            "place_ids": place_ids,
            "landmark_tokens": landmark_tokens_tensor,
            "event_tokens": self._normalize_feature_matrix(
                torch.stack(event_tokens, dim=0) if event_tokens else place_tokens.new_zeros((0, place_tokens.shape[-1])),
                name="retrieve_global_graph_tokens.event_tokens",
            ),
            "retrieved_landmark_count": float(landmark_tokens_tensor.shape[0]),
            "landmark_token_norm_mean": self._last_retrieved_landmark_norm_mean,
            "landmark_gate_avg": self._last_retrieved_landmark_gate_avg,
            "retrieval_scores": retrieval_scores,
            "retrieval_goal_raw_avg": self._mean_float_or_nan(all_goal_raw),
            "retrieval_goal_norm_avg": self._mean_float_or_nan(all_goal_norm),
            "retrieval_goal_norm_min": min(all_goal_norm) if all_goal_norm else float("nan"),
            "retrieval_goal_norm_max": max(all_goal_norm) if all_goal_norm else float("nan"),
            "topk_goal_norm_mean": self._mean_float_or_nan(selected_goal_norm),
            "retrieval_topk_goal_norm": self._mean_float_or_nan(selected_goal_norm),
            "retrieval_non_topk_goal_norm": self._mean_float_or_nan(non_topk_goal_norm),
            "retrieval_goal_component_avg": self._mean_float_or_nan(all_goal_components),
            "retrieval_visual_component_avg": self._mean_float_or_nan(all_visual_components),
            "retrieval_visit_component_avg": self._mean_float_or_nan(all_visit_components),
            "topk_goal_component_avg": self._mean_float_or_nan(selected_goal_components),
            "topk_visual_component_avg": self._mean_float_or_nan(selected_visual_components),
            "topk_visit_component_avg": self._mean_float_or_nan(selected_visit_components),
            "non_topk_goal_component_avg": self._mean_float_or_nan(non_topk_goal_components),
            "non_topk_visual_component_avg": self._mean_float_or_nan(non_topk_visual_components),
            "non_topk_visit_component_avg": self._mean_float_or_nan(non_topk_visit_components),
            "topk_goal_largest_component_rate": float(selected_largest_components.count("goal")) / selected_largest_denominator,
            "topk_visual_largest_component_rate": float(selected_largest_components.count("visual")) / selected_largest_denominator,
            "topk_visit_largest_component_rate": float(selected_largest_components.count("visit")) / selected_largest_denominator,
        }

    def retrieve_local_subgraph_tokens(
        self,
        active_node_id: Optional[int],
        instruction_feat: torch.Tensor,
        hop: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Retrieve local active-subgraph tokens for fine refinement."""

        self._require_started()
        device = self._resolve_device(ref_tensor=instruction_feat, device=device)
        hidden_dtype = self._resolve_dtype(instruction_feat, dtype=dtype)
        self.token_encoder.to(device=device, dtype=hidden_dtype)
        if not self.place_nodes:
            zero = self._normalize_feature_matrix(
                self._to_device(self.fallback_feature, device, hidden_dtype),
                name="retrieve_local_subgraph_tokens.fallback_local_tokens",
            )
            return {
                "active_place_token": zero,
                "local_tokens": zero,
                "local_context": zero.squeeze(0),
                "patch_summary": zero.squeeze(0),
                "local_place_count": 0.0,
                "local_token_count": 0.0,
                "active_node_valid": 0.0,
            }

        valid_ids = {place.node_id for place in self.place_nodes}
        active_node_valid = 1.0
        if active_node_id is None:
            active_node_id = self.place_nodes[0].node_id
            active_node_valid = 0.0
        elif active_node_id not in valid_ids:
            active_node_id = self.place_nodes[0].node_id
            active_node_valid = 0.0

        connected_ids = {active_node_id}
        frontier = {active_node_id}
        for _ in range(max(int(hop), 1)):
            next_frontier = set()
            for edge_list in (self.temporal_edges, self.spatial_edges, self.reobservation_edges):
                for src, dst in edge_list:
                    if src in frontier and dst not in connected_ids:
                        connected_ids.add(dst)
                        next_frontier.add(dst)
                    if dst in frontier and src not in connected_ids:
                        connected_ids.add(src)
                        next_frontier.add(src)
            frontier = next_frontier

        ordered_ids = sorted(connected_ids)
        ordered_places = [self._place_by_id(node_id) for node_id in ordered_ids]
        place_tokens = self._normalize_feature_matrix(
            torch.stack(
                [self.token_encoder.encode_place(place, device=device, dtype=hidden_dtype) for place in ordered_places],
                dim=0,
            ),
            name="retrieve_local_subgraph_tokens.place_tokens",
        )
        patch_summaries = [
            self._mean_tensor_on_device(
                place.recent_patch_bank,
                place.visual_prototype(),
                device,
                hidden_dtype,
            )
            for place in ordered_places
        ]
        patch_summary = self._mean_tensor_on_device(
            patch_summaries,
            self.fallback_feature,
            device,
            hidden_dtype,
        )

        if bool(getattr(self.args, "landmark_as_auxiliary", True)):
            selected_local_landmarks = self._select_landmarks_for_places(
                [active_node_id],
                limit=int(getattr(self.args, "landmark_retrieve_k", 2)),
                active_only=True,
            )
        else:
            selected_local_landmarks = self._select_landmarks_for_places(
                ordered_ids,
                limit=int(getattr(self.args, "landmark_retrieve_k", 2)),
                active_only=False,
            )
        landmark_tokens_tensor = self._normalize_feature_matrix(
            self._encode_landmark_tokens(selected_local_landmarks, device, hidden_dtype, place_tokens),
            name="retrieve_local_subgraph_tokens.landmark_tokens",
        )
        ref_step = max(place.last_seen_step for place in self.place_nodes)
        event_tokens = [
            self.token_encoder.encode_event(event, ref_step, hidden_dtype, device)
            for event in self.event_nodes
            if event.attached_place_id in connected_ids
        ]
        local_tokens = [place_tokens]
        if landmark_tokens_tensor.shape[0] > 0:
            local_tokens.append(landmark_tokens_tensor)
        if event_tokens:
            local_tokens.append(
                self._normalize_feature_matrix(
                    torch.stack(event_tokens, dim=0),
                    name="retrieve_local_subgraph_tokens.event_tokens",
                )
            )
        local_token_shapes = {f"local_tokens[{idx}]": tuple(token.shape) for idx, token in enumerate(local_tokens)}
        if any(token.dim() != 2 for token in local_tokens):
            raise ValueError(f"retrieve_local_subgraph_tokens expected all local tokens to be [N, D], got {local_token_shapes}")
        local_tokens_cat = torch.cat(local_tokens, dim=0)
        return {
            "active_place_token": self._normalize_token_2d(
                self.token_encoder.encode_place(
                    self._place_by_id(active_node_id),
                    device=device,
                    dtype=hidden_dtype,
                ),
                name="retrieve_local_subgraph_tokens.active_place_token",
            ),
            "local_tokens": local_tokens_cat,
            "local_context": local_tokens_cat.mean(dim=0),
            "patch_summary": patch_summary,
            "local_place_count": float(len(ordered_places)),
            "local_token_count": float(local_tokens_cat.shape[0]),
            "local_landmark_count": float(landmark_tokens_tensor.shape[0]),
            "active_node_valid": float(active_node_valid),
        }

    def _landmark_debug_stats(self, place_token_count: int = 0) -> Dict[str, float]:
        """Summarize landmark filtering, retrieval, attachment density, and token scale."""

        if not self._landmarks_enabled():
            return {}
        degrees = [len(landmark.attached_place_ids) for landmark in self.landmark_nodes]
        confidences = [float(landmark.confidence) for landmark in self.landmark_nodes]
        attached_edges = float(sum(degrees))
        place_count = max(len(self.place_nodes), 1)
        landmark_count = max(len(self.landmark_nodes), 1)
        all_to_all = bool(
            self.landmark_nodes
            and len(self.place_nodes) > 1
            and attached_edges >= float(len(self.landmark_nodes) * len(self.place_nodes))
        )
        retrieved = float(self._last_retrieved_landmark_count)
        return {
            "raw_landmark_count": float(self._raw_landmark_count),
            "valid_landmark_count": float(len(self.landmark_nodes)),
            "retrieved_landmark_count": retrieved,
            "attached_landmark_edges": attached_edges,
            "avg_landmark_degree": self._mean_float([float(value) for value in degrees]),
            "max_landmark_degree": float(max(degrees) if degrees else 0.0),
            "landmark_conf_avg": self._mean_float(confidences),
            "landmark_conf_min": float(min(confidences) if confidences else 0.0),
            "landmark_conf_max": float(max(confidences) if confidences else 0.0),
            "landmark_conf_range": float((max(confidences) - min(confidences)) if confidences else 0.0),
            "landmark_text_rel_avg": self._mean_float(self._landmark_text_rel_values),
            "landmark_geo_score_avg": self._mean_float(self._landmark_geo_score_values),
            "landmark_visual_support_avg": self._mean_float(self._landmark_visual_support_values),
            "landmark_token_norm_mean": float(self._last_retrieved_landmark_norm_mean),
            "landmark_gate_avg": float(self._last_retrieved_landmark_gate_avg),
            "landmark_place_token_ratio": retrieved / max(float(place_token_count), 1.0),
            "landmark_empty_ratio": 1.0 if retrieved <= 0.0 else 0.0,
            "landmark_filtered_low_conf_count": float(self._filtered_low_conf_count),
            "landmark_all_to_all_detected": float(all_to_all),
            "landmark_degree_budget": float(getattr(self.args, "max_landmark_degree", 3)),
            "landmark_place_degree_budget": float(getattr(self.args, "landmark_retrieve_k", 2)),
            "landmark_node_density": float(len(self.landmark_nodes)) / float(place_count),
            "landmark_per_place_budget_ratio": attached_edges / float(place_count * landmark_count),
        }

    def export_for_transformer(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, object]:
        """Export graph retrieval outputs in ET-compatible tensor form."""

        self._require_started()
        device = self._resolve_device(ref_tensor=self.instruction_feat, device=device)
        hidden_dtype = self._resolve_dtype(self.instruction_feat, dtype=dtype)
        self.token_encoder.to(device=device, dtype=hidden_dtype)
        if not self.place_nodes:
            max_neighbors = max(int(self.args.patch_bank_size), 1)
            zero_feature = self._normalize_feature_matrix(
                self._to_device(self.fallback_feature, device, hidden_dtype),
                name="export_for_transformer.node_features[fallback]",
            )
            empty_tokens = self._normalize_feature_matrix(
                zero_feature.new_zeros((0, zero_feature.shape[-1])),
                name="export_for_transformer.global_local_tokens[empty]",
            )
            base_positions = self._to_device(self.base_positions, device, self.base_positions.dtype)
            return {
                "node_features": self._normalize_feature_matrix(
                    zero_feature,
                    name="export_for_transformer.node_features[fallback]",
                ),
                "node_positions": base_positions[:1].clone(),
                "node_patch": self._normalize_feature_matrix(
                    zero_feature.clone(),
                    name="export_for_transformer.node_patch[fallback]",
                ),
                "cell_to_node_map": torch.zeros(self.base_positions.shape[0], device=device, dtype=torch.long),
                "neighbor_index": torch.zeros(1, max_neighbors, device=device, dtype=torch.long),
                "neighbor_mask": torch.ones(1, max_neighbors, device=device, dtype=torch.bool),
                "local_context": self._to_device(self.fallback_feature.clone(), device, hidden_dtype),
                "local_patch_context": self._to_device(self.fallback_feature.clone(), device, hidden_dtype),
                "global_retrieved_tokens": empty_tokens,
                "local_subgraph_tokens": empty_tokens,
                "stats": {
                    "place_node_count": 0.0,
                    "avg_place_nodes": 0.0,
                    "max_place_nodes_used": 0.0,
                    "min_place_nodes_used": 0.0,
                    "nodes_after_merge": 0.0,
                    "created_nodes": 0.0,
                    "updated_nodes": 0.0,
                    "merged_nodes": 0.0,
                    "create_place_nodes_count": 0.0,
                    "merge_place_nodes_count": 0.0,
                    "update_existing_place_nodes_count": 0.0,
                    "landmark_nodes": 0.0,
                    "event_nodes": 0.0,
                    "temporal_edges": 0.0,
                    "spatial_edges": 0.0,
                    "semantic_edges": 0.0,
                    "reobservation_edges": 0.0,
                    "active_place_id": -1.0,
                    "global_retrieved_nodes": 0.0,
                    "local_retrieved_nodes": 0.0,
                    "active_node_valid_ratio": 0.0,
                    "empty_retrieval_ratio": 1.0,
                    "node_saturation_ratio": 0.0,
                    "retrieval_coverage": 0.0,
                    "avg_goal_relevance": 0.0,
                    "max_goal_relevance": 0.0,
                    "avg_goal_relevance_norm": float("nan"),
                    "max_goal_relevance_norm": float("nan"),
                    "created_goal_relevance": float("nan"),
                    "updated_goal_relevance": float("nan"),
                    "merged_goal_relevance": float("nan"),
                    "created_goal_relevance_norm": float("nan"),
                    "updated_goal_relevance_norm": float("nan"),
                    "merged_goal_relevance_norm": float("nan"),
                    "created_goal_raw": float("nan"),
                    "updated_goal_raw": float("nan"),
                    "merged_goal_raw": float("nan"),
                    "created_goal_norm": float("nan"),
                    "updated_goal_norm": float("nan"),
                    "merged_goal_norm": float("nan"),
                    "retrieval_goal_raw_avg": float("nan"),
                    "retrieval_goal_norm_avg": float("nan"),
                    "retrieval_goal_norm_min": float("nan"),
                    "retrieval_goal_norm_max": float("nan"),
                    "retrieval_topk_goal_norm": float("nan"),
                    "retrieval_non_topk_goal_norm": float("nan"),
                    "topk_goal_norm_mean": float("nan"),
                    "retrieval_goal_component_avg": float("nan"),
                    "retrieval_visual_component_avg": float("nan"),
                    "retrieval_visit_component_avg": float("nan"),
                    "topk_goal_component_avg": float("nan"),
                    "topk_visual_component_avg": float("nan"),
                    "topk_visit_component_avg": float("nan"),
                    "non_topk_goal_component_avg": float("nan"),
                    "non_topk_visual_component_avg": float("nan"),
                    "non_topk_visit_component_avg": float("nan"),
                    "topk_goal_largest_component_rate": float("nan"),
                    "topk_visual_largest_component_rate": float("nan"),
                    "topk_visit_largest_component_rate": float("nan"),
                    "global_token_count": 0.0,
                    "local_token_count": 0.0,
                    "topo_token_norm_mean": 0.0,
                    "topo_token_norm_std": 0.0,
                    "global_token_norm_mean": 0.0,
                    "local_token_norm_mean": 0.0,
                },
            }

        global_out = self.retrieve_global_graph_tokens(
            self.instruction_feat,
            int(self.args.global_retrieve_k),
            current_feature=self.fallback_feature,
            device=device,
            dtype=hidden_dtype,
        )
        active_place_id = self._active_place_id
        if active_place_id is None and len(self.place_nodes) > 0:
            active_place_id = self.place_nodes[0].node_id
        local_out = self.retrieve_local_subgraph_tokens(
            active_place_id,
            self.instruction_feat,
            hop=int(self.args.local_hops),
            device=device,
            dtype=hidden_dtype,
        )

        place_ids = global_out["place_ids"]
        place_tokens = global_out["place_tokens"]
        place_positions = global_out["place_positions"]
        node_patch = self._normalize_feature_matrix(
            torch.stack(
                [
                    self._mean_tensor_on_device(
                        self._place_by_id(int(place_id.item())).recent_patch_bank,
                        self._place_by_id(int(place_id.item())).visual_prototype(),
                        device,
                        hidden_dtype,
                    )
                    for place_id in place_ids
                ],
                dim=0,
            ),
            name="export_for_transformer.node_patch",
        )

        max_neighbors = max(int(self.args.patch_bank_size), 1)
        neighbor_index = torch.zeros(
            place_tokens.shape[0],
            max_neighbors,
            device=place_tokens.device,
            dtype=torch.long,
        )
        neighbor_mask = torch.ones(
            place_tokens.shape[0],
            max_neighbors,
            device=place_tokens.device,
            dtype=torch.bool,
        )
        place_id_to_slot = {int(place_id.item()): slot for slot, place_id in enumerate(place_ids)}
        edge_lists = (self.temporal_edges, self.spatial_edges, self.reobservation_edges)
        for edge_list in edge_lists:
            for src, dst in edge_list:
                if src not in place_id_to_slot or dst not in place_id_to_slot:
                    continue
                src_slot = place_id_to_slot[src]
                insert_slots = torch.nonzero(neighbor_mask[src_slot], as_tuple=False).flatten()
                if insert_slots.numel() == 0:
                    continue
                first_slot = int(insert_slots[0].item())
                neighbor_index[src_slot, first_slot] = place_id_to_slot[dst]
                neighbor_mask[src_slot, first_slot] = False

        if self.place_nodes:
            all_positions = torch.stack(
                [self._to_device(place.center_xy, device, self.base_positions.dtype) for place in self.place_nodes],
                dim=0,
            )
            all_ids = [place.node_id for place in self.place_nodes]
            base_positions = self._to_device(self.base_positions, device, self.base_positions.dtype)
            nearest_slot = torch.argmin(torch.cdist(base_positions, all_positions), dim=1)
            cell_to_node_map = torch.zeros(
                self.base_positions.shape[0],
                device=device,
                dtype=torch.long,
            )
            for cell_idx in range(nearest_slot.shape[0]):
                raw_place_id = all_ids[int(nearest_slot[cell_idx].item())]
                cell_to_node_map[cell_idx] = place_id_to_slot.get(raw_place_id, 0)
        else:
            cell_to_node_map = torch.zeros(self.base_positions.shape[0], device=device, dtype=torch.long)

        stats = {
            "place_node_count": float(len(self.place_nodes)),
            "avg_place_nodes": float(len(self.place_nodes)),
            "max_place_nodes_used": float(len(self.place_nodes)),
            "min_place_nodes_used": float(len(self.place_nodes)),
            "nodes_after_merge": float(len(self.place_nodes)),
            "created_nodes": float(self._step_new_nodes),
            "updated_nodes": float(self._step_updated_nodes),
            "merged_nodes": float(self._step_merged_nodes),
            "create_place_nodes_count": float(self._step_new_nodes),
            "merge_place_nodes_count": float(self._step_merged_nodes),
            "update_existing_place_nodes_count": float(self._step_updated_nodes),
            "landmark_nodes": float(len(self.landmark_nodes)),
            "event_nodes": float(len(self.event_nodes)),
            "temporal_edges": float(len(self.temporal_edges)),
            "spatial_edges": float(len(self.spatial_edges)),
            "semantic_edges": float(len(self.semantic_edges)),
            "reobservation_edges": float(len(self.reobservation_edges)),
            "active_place_id": float(self._active_place_id) if self._active_place_id is not None else -1.0,
        }
        goal_relevances = [float(place.goal_relevance) for place in self.place_nodes]
        goal_relevances_norm = [float(place.goal_relevance_norm) for place in self.place_nodes]
        local_tokens = local_out["local_tokens"]
        global_norm_stats = self._token_norm_stats(place_tokens)
        local_norm_stats = self._token_norm_stats(local_tokens)
        if place_tokens.numel() > 0 and local_tokens.numel() > 0:
            topo_tokens_for_stats = torch.cat((place_tokens, local_tokens), dim=0)
        elif place_tokens.numel() > 0:
            topo_tokens_for_stats = place_tokens
        else:
            topo_tokens_for_stats = local_tokens
        topo_norm_stats = self._token_norm_stats(topo_tokens_for_stats)
        stats.update(
            {
                "global_retrieved_nodes": float(place_tokens.shape[0]),
                "local_retrieved_nodes": float(local_out.get("local_place_count", 0.0)),
                "local_retrieved_landmark_count": float(local_out.get("local_landmark_count", 0.0)),
                "active_node_valid_ratio": float(local_out.get("active_node_valid", 0.0)),
                "empty_retrieval_ratio": 0.0 if place_tokens.shape[0] > 0 else 1.0,
                "node_saturation_ratio": float(len(self.place_nodes)) / max(float(self.args.max_place_nodes), 1.0),
                "retrieval_coverage": float(place_tokens.shape[0]) / max(float(len(self.place_nodes)), 1.0),
                "avg_goal_relevance": self._mean_float(goal_relevances),
                "max_goal_relevance": max(goal_relevances) if goal_relevances else 0.0,
                "avg_goal_relevance_norm": self._mean_float_or_nan(goal_relevances_norm),
                "max_goal_relevance_norm": max(goal_relevances_norm) if goal_relevances_norm else float("nan"),
                "created_goal_relevance": self._mean_float_or_nan(self._step_created_goal_relevance),
                "updated_goal_relevance": self._mean_float_or_nan(self._step_updated_goal_relevance),
                "merged_goal_relevance": self._mean_float_or_nan(self._step_merged_goal_relevance),
                "created_goal_relevance_norm": self._mean_float_or_nan(self._step_created_goal_relevance_norm),
                "updated_goal_relevance_norm": self._mean_float_or_nan(self._step_updated_goal_relevance_norm),
                "merged_goal_relevance_norm": self._mean_float_or_nan(self._step_merged_goal_relevance_norm),
                "created_goal_raw": self._mean_float_or_nan(self._step_created_goal_relevance),
                "updated_goal_raw": self._mean_float_or_nan(self._step_updated_goal_relevance),
                "merged_goal_raw": self._mean_float_or_nan(self._step_merged_goal_relevance),
                "created_goal_norm": self._mean_float_or_nan(self._step_created_goal_relevance_norm),
                "updated_goal_norm": self._mean_float_or_nan(self._step_updated_goal_relevance_norm),
                "merged_goal_norm": self._mean_float_or_nan(self._step_merged_goal_relevance_norm),
                "retrieval_goal_raw_avg": float(global_out.get("retrieval_goal_raw_avg", float("nan"))),
                "retrieval_goal_norm_avg": float(global_out.get("retrieval_goal_norm_avg", float("nan"))),
                "retrieval_goal_norm_min": float(global_out.get("retrieval_goal_norm_min", float("nan"))),
                "retrieval_goal_norm_max": float(global_out.get("retrieval_goal_norm_max", float("nan"))),
                "retrieval_topk_goal_norm": float(global_out.get("retrieval_topk_goal_norm", float("nan"))),
                "retrieval_non_topk_goal_norm": float(global_out.get("retrieval_non_topk_goal_norm", float("nan"))),
                "topk_goal_norm_mean": float(global_out.get("topk_goal_norm_mean", float("nan"))),
                "retrieval_goal_component_avg": float(global_out.get("retrieval_goal_component_avg", float("nan"))),
                "retrieval_visual_component_avg": float(global_out.get("retrieval_visual_component_avg", float("nan"))),
                "retrieval_visit_component_avg": float(global_out.get("retrieval_visit_component_avg", float("nan"))),
                "topk_goal_component_avg": float(global_out.get("topk_goal_component_avg", float("nan"))),
                "topk_visual_component_avg": float(global_out.get("topk_visual_component_avg", float("nan"))),
                "topk_visit_component_avg": float(global_out.get("topk_visit_component_avg", float("nan"))),
                "non_topk_goal_component_avg": float(global_out.get("non_topk_goal_component_avg", float("nan"))),
                "non_topk_visual_component_avg": float(global_out.get("non_topk_visual_component_avg", float("nan"))),
                "non_topk_visit_component_avg": float(global_out.get("non_topk_visit_component_avg", float("nan"))),
                "topk_goal_largest_component_rate": float(global_out.get("topk_goal_largest_component_rate", float("nan"))),
                "topk_visual_largest_component_rate": float(global_out.get("topk_visual_largest_component_rate", float("nan"))),
                "topk_visit_largest_component_rate": float(global_out.get("topk_visit_largest_component_rate", float("nan"))),
                "global_token_count": float(place_tokens.shape[0]),
                "local_token_count": float(local_tokens.shape[0]),
                "topo_token_norm_mean": topo_norm_stats["mean"],
                "topo_token_norm_std": topo_norm_stats["std"],
                "global_token_norm_mean": global_norm_stats["mean"],
                "local_token_norm_mean": local_norm_stats["mean"],
            }
        )
        stats.update(
            {
                "retrieved_landmark_count": float(global_out.get("retrieved_landmark_count", 0.0)),
                "landmark_token_norm_mean": float(global_out.get("landmark_token_norm_mean", 0.0)),
                "landmark_gate_avg": float(global_out.get("landmark_gate_avg", 0.0)),
            }
        )
        stats.update(self._landmark_debug_stats(place_token_count=int(place_tokens.shape[0])))
        stats.update(self._last_update_stats)
        return {
            "node_features": self._normalize_feature_matrix(
                place_tokens,
                name="export_for_transformer.node_features",
            ),
            "node_positions": place_positions,
            "node_patch": self._normalize_feature_matrix(
                node_patch,
                name="export_for_transformer.node_patch",
            ),
            "cell_to_node_map": cell_to_node_map,
            "neighbor_index": neighbor_index,
            "neighbor_mask": neighbor_mask,
            "local_context": local_out["local_context"],
            "local_patch_context": local_out["patch_summary"],
            "global_retrieved_tokens": place_tokens,
            "local_subgraph_tokens": local_tokens,
            "stats": stats,
        }


class TopoMemoryBuilder(nn.Module):
    """Fallback rebuild path kept for debugging and compatibility."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.token_encoder = GraphTokenEncoder(args.demb)

    def build_single(
        self,
        history_features: torch.Tensor,
        history_cell_ids: torch.Tensor,
        history_times: torch.Tensor,
        history_xy: torch.Tensor,
        landmark_history: Optional[Sequence[Sequence[dict]]],
        base_positions: torch.Tensor,
        lang_goal_embed: torch.Tensor,
        current_grid: int,
        fallback_feature: torch.Tensor,
        time_decay_rate: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        """Rebuild a graph by replaying history as a debug / fallback path."""

        graph = TopoMemoryGraph(self.args, self.token_encoder)
        graph.start_episode(base_positions, lang_goal_embed, fallback_feature)
        graph.build_from_history(
            history_features=history_features,
            history_cell_ids=history_cell_ids,
            history_times=history_times,
            history_xy=history_xy,
            landmark_history=landmark_history,
            current_grid=current_grid,
            time_decay_rate=time_decay_rate,
        )
        return graph.export_for_transformer(
            device=lang_goal_embed.device,
            dtype=lang_goal_embed.dtype,
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_cell_ids: torch.Tensor,
        history_times: torch.Tensor,
        history_xy: torch.Tensor,
        landmark_history: Optional[Sequence[Sequence[Sequence[dict]]]],
        base_positions: torch.Tensor,
        lang_goal_embed: torch.Tensor,
        current_grids: torch.Tensor,
        fallback_features: torch.Tensor,
        time_decay_rate: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Batch the rebuild-from-history fallback outputs."""

        outputs = []
        max_nodes = 1
        max_local_tokens = 1
        max_neighbors = max(int(self.args.patch_bank_size), 1)
        for batch_idx in range(history_features.shape[0]):
            landmarks_i = None if landmark_history is None else landmark_history[batch_idx]
            output = self.build_single(
                history_features[batch_idx],
                history_cell_ids[batch_idx].long(),
                history_times[batch_idx],
                history_xy[batch_idx],
                landmarks_i,
                base_positions[batch_idx],
                lang_goal_embed[batch_idx],
                int(current_grids[batch_idx].item()),
                fallback_features[batch_idx],
                time_decay_rate,
            )
            outputs.append(output)
            max_nodes = max(max_nodes, int(output["node_features"].shape[0]))
            max_local_tokens = max(max_local_tokens, int(output["local_subgraph_tokens"].shape[0]))

        node_features_padded = history_features.new_zeros(history_features.shape[0], max_nodes, history_features.shape[-1])
        node_positions_padded = base_positions.new_zeros(history_features.shape[0], max_nodes, 2)
        node_patch_padded = history_features.new_zeros(history_features.shape[0], max_nodes, history_features.shape[-1])
        node_padding_mask = torch.ones(history_features.shape[0], max_nodes, device=history_features.device, dtype=torch.bool)
        local_subgraph_tokens_padded = history_features.new_zeros(history_features.shape[0], max_local_tokens, history_features.shape[-1])
        local_token_padding_mask = torch.ones(history_features.shape[0], max_local_tokens, device=history_features.device, dtype=torch.bool)
        cell_to_node_map = torch.zeros(history_features.shape[0], base_positions.shape[1], device=history_features.device, dtype=torch.long)
        neighbor_index_padded = torch.zeros(history_features.shape[0], max_nodes, max_neighbors, device=history_features.device, dtype=torch.long)
        neighbor_mask_padded = torch.ones(history_features.shape[0], max_nodes, max_neighbors, device=history_features.device, dtype=torch.bool)
        local_context = history_features.new_zeros(history_features.shape[0], history_features.shape[-1])
        local_patch_context = history_features.new_zeros(history_features.shape[0], history_features.shape[-1])
        stats_accumulator: Dict[str, List[float]] = {}
        place_node_counts: List[float] = []
        global_retrieved_counts: List[float] = []
        local_retrieved_counts: List[float] = []
        active_valid_values: List[float] = []
        empty_retrieval_values: List[float] = []

        for batch_idx, output in enumerate(outputs):
            count = int(output["node_features"].shape[0])
            node_features_padded[batch_idx, :count] = output["node_features"]
            node_positions_padded[batch_idx, :count] = output["node_positions"]
            node_patch_padded[batch_idx, :count] = output["node_patch"]
            node_padding_mask[batch_idx, :count] = False
            local_tokens = output["local_subgraph_tokens"]
            local_count = int(local_tokens.shape[0])
            if local_count > 0:
                local_subgraph_tokens_padded[batch_idx, :local_count] = local_tokens
                local_token_padding_mask[batch_idx, :local_count] = False
            cell_to_node_map[batch_idx] = output["cell_to_node_map"]
            neighbor_index = output["neighbor_index"]
            neighbor_mask = output["neighbor_mask"]
            neighbor_index_padded[batch_idx, :count, : neighbor_index.shape[1]] = neighbor_index
            neighbor_mask_padded[batch_idx, :count, : neighbor_mask.shape[1]] = neighbor_mask
            local_context[batch_idx] = output["local_context"]
            local_patch_context[batch_idx] = output["local_patch_context"]
            for key, value in output["stats"].items():
                stats_accumulator.setdefault(key, []).append(float(value))
            stats = output["stats"]
            place_node_counts.append(float(stats.get("place_node_count", stats.get("nodes_after_merge", 0.0))))
            global_retrieved_counts.append(float(stats.get("global_retrieved_nodes", 0.0)))
            local_retrieved_counts.append(float(stats.get("local_retrieved_nodes", 0.0)))
            active_valid_values.append(float(stats.get("active_node_valid_ratio", 0.0)))
            empty_retrieval_values.append(float(stats.get("empty_retrieval_ratio", 0.0)))

        mean_stats = {}
        for key, values in stats_accumulator.items():
            finite_values = [value for value in values if math.isfinite(value)]
            mean_stats[key] = (
                float(sum(finite_values) / len(finite_values))
                if finite_values else float("nan")
            )
        if place_node_counts:
            mean_stats.update(
                {
                    "avg_place_nodes": float(sum(place_node_counts) / len(place_node_counts)),
                    "max_place_nodes_used": float(max(place_node_counts)),
                    "min_place_nodes_used": float(min(place_node_counts)),
                    "global_retrieved_nodes": float(sum(global_retrieved_counts) / len(global_retrieved_counts)),
                    "local_retrieved_nodes": float(sum(local_retrieved_counts) / len(local_retrieved_counts)),
                    "active_node_valid_ratio": float(sum(active_valid_values) / len(active_valid_values)),
                    "empty_retrieval_ratio": float(sum(empty_retrieval_values) / len(empty_retrieval_values)),
                }
            )
        return {
            "node_features_padded": node_features_padded,
            "global_retrieved_tokens_padded": node_features_padded,
            "node_positions_padded": node_positions_padded,
            "node_patch_padded": node_patch_padded,
            "node_padding_mask": node_padding_mask,
            "local_subgraph_tokens_padded": local_subgraph_tokens_padded,
            "local_token_padding_mask": local_token_padding_mask,
            "cell_to_node_map": cell_to_node_map,
            "neighbor_index_padded": neighbor_index_padded,
            "neighbor_mask_padded": neighbor_mask_padded,
            "local_context": local_context,
            "local_patch_context": local_patch_context,
            "stats": mean_stats,
        }


class BatchedTopoMemory:
    """Maintain one persistent topo graph per active batch element."""

    def __init__(self, args):
        self.args = args
        self.token_encoder = GraphTokenEncoder(args.demb)
        self.graphs: List[TopoMemoryGraph] = []

    def reset_all(self) -> None:
        """Drop all env graphs."""

        self.graphs = []

    def reset_env(self, env_idx: int) -> None:
        """Reset the graph for a single env slot."""

        if env_idx < len(self.graphs):
            graph = self.graphs[env_idx]
            if graph.base_positions is not None and graph.instruction_feat is not None and graph.fallback_feature is not None:
                base_positions = graph.base_positions.clone()
                instruction_feat = graph.instruction_feat.clone()
                fallback_feature = graph.fallback_feature.clone()
                graph.start_episode(base_positions, instruction_feat, fallback_feature)
            else:
                graph.reset()

    def start_batch(
        self,
        base_positions: torch.Tensor,
        instruction_feat: torch.Tensor,
        fallback_features: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize a fresh persistent graph for each env in the batch."""

        self.graphs = []
        if fallback_features is None:
            fallback_features = instruction_feat.new_zeros(instruction_feat.shape[0], instruction_feat.shape[-1])
        for env_idx in range(base_positions.shape[0]):
            graph = TopoMemoryGraph(self.args, self.token_encoder)
            graph.start_episode(
                base_positions=base_positions[env_idx],
                instruction_feat=instruction_feat[env_idx],
                fallback_feature=fallback_features[env_idx],
            )
            self.graphs.append(graph)

    def ensure_started(
        self,
        base_positions: torch.Tensor,
        instruction_feat: torch.Tensor,
        fallback_features: Optional[torch.Tensor] = None,
    ) -> None:
        """Start graphs lazily if the manager is empty or mismatched."""

        if len(self.graphs) != base_positions.shape[0]:
            self.start_batch(base_positions, instruction_feat, fallback_features)

    def update_env_step(
        self,
        env_idx: int,
        observation_feature: torch.Tensor,
        cell_id: int,
        xy: Optional[torch.Tensor],
        landmark_info: Sequence[dict],
        step_id: int,
        time_decay_rate: Optional[torch.Tensor],
        fallback_feature: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Update one env graph in-place."""

        graph = self.graphs[env_idx]
        if fallback_feature is not None:
            graph.fallback_feature = fallback_feature.detach().clone()
        return graph.update_step(
            observation_feature=observation_feature,
            cell_id=cell_id,
            xy=xy,
            landmark_info=landmark_info,
            step_id=step_id,
            time_decay_rate=time_decay_rate,
        )

    def retrieve_batch(self, template_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Collect padded retrieval tensors from all active graphs."""

        if not self.graphs:
            raise RuntimeError("BatchedTopoMemory.start_batch(...) must run before retrieve_batch().")

        device = template_tensor.device
        dtype = template_tensor.dtype
        self.token_encoder.to(device=device, dtype=dtype)
        outputs = [graph.export_for_transformer(device=device, dtype=dtype) for graph in self.graphs]
        max_nodes = max(int(output["node_features"].shape[0]) for output in outputs)
        max_local_tokens = max(max(int(output["local_subgraph_tokens"].shape[0]), 1) for output in outputs)
        max_neighbors = max(int(output["neighbor_index"].shape[1]) for output in outputs)
        batch_size = len(outputs)
        hidden_size = template_tensor.shape[-1]

        node_features_padded = template_tensor.new_zeros(batch_size, max_nodes, hidden_size)
        node_positions_padded = template_tensor.new_zeros(batch_size, max_nodes, 2)
        node_patch_padded = template_tensor.new_zeros(batch_size, max_nodes, hidden_size)
        node_padding_mask = torch.ones(batch_size, max_nodes, device=template_tensor.device, dtype=torch.bool)
        local_subgraph_tokens_padded = template_tensor.new_zeros(batch_size, max_local_tokens, hidden_size)
        local_token_padding_mask = torch.ones(batch_size, max_local_tokens, device=template_tensor.device, dtype=torch.bool)
        cell_to_node_map = torch.zeros(batch_size, self.graphs[0].base_positions.shape[0], device=template_tensor.device, dtype=torch.long)
        neighbor_index_padded = torch.zeros(batch_size, max_nodes, max_neighbors, device=template_tensor.device, dtype=torch.long)
        neighbor_mask_padded = torch.ones(batch_size, max_nodes, max_neighbors, device=template_tensor.device, dtype=torch.bool)
        local_context = template_tensor.new_zeros(batch_size, hidden_size)
        local_patch_context = template_tensor.new_zeros(batch_size, hidden_size)
        stats_accumulator: Dict[str, List[float]] = {}
        place_node_counts: List[float] = []
        global_retrieved_counts: List[float] = []
        local_retrieved_counts: List[float] = []
        active_valid_values: List[float] = []
        empty_retrieval_values: List[float] = []

        for env_idx, output in enumerate(outputs):
            node_features = self.graphs[env_idx]._normalize_feature_matrix(
                output["node_features"],
                name=f"BatchedTopoMemory.retrieve_batch.output[{env_idx}].node_features",
            )
            if node_features.dim() != 2:
                raise RuntimeError(
                    f"BatchedTopoMemory.retrieve_batch expected rank-2 node_features for env {env_idx}, "
                    f"got shape {tuple(node_features.shape)}"
                )
            node_patch = self.graphs[env_idx]._normalize_feature_matrix(
                output["node_patch"],
                name=f"BatchedTopoMemory.retrieve_batch.output[{env_idx}].node_patch",
            )
            count = int(node_features.shape[0])
            node_features_padded[env_idx, :count] = node_features.to(template_tensor.device)
            node_positions_padded[env_idx, :count] = output["node_positions"].to(template_tensor.device)
            node_patch_padded[env_idx, :count] = node_patch.to(template_tensor.device)
            node_padding_mask[env_idx, :count] = False
            local_tokens = output["local_subgraph_tokens"].to(template_tensor.device)
            local_count = int(local_tokens.shape[0])
            if local_count > 0:
                local_subgraph_tokens_padded[env_idx, :local_count] = local_tokens
                local_token_padding_mask[env_idx, :local_count] = False
            cell_to_node_map[env_idx] = output["cell_to_node_map"].to(template_tensor.device)
            neighbor_index = output["neighbor_index"].to(template_tensor.device)
            neighbor_mask = output["neighbor_mask"].to(template_tensor.device)
            neighbor_index_padded[env_idx, :count, : neighbor_index.shape[1]] = neighbor_index
            neighbor_mask_padded[env_idx, :count, : neighbor_mask.shape[1]] = neighbor_mask
            local_context[env_idx] = output["local_context"].to(template_tensor.device)
            local_patch_context[env_idx] = output["local_patch_context"].to(template_tensor.device)
            for key, value in output["stats"].items():
                stats_accumulator.setdefault(key, []).append(float(value))
            stats = output["stats"]
            place_node_counts.append(float(stats.get("place_node_count", stats.get("nodes_after_merge", 0.0))))
            global_retrieved_counts.append(float(stats.get("global_retrieved_nodes", 0.0)))
            local_retrieved_counts.append(float(stats.get("local_retrieved_nodes", 0.0)))
            active_valid_values.append(float(stats.get("active_node_valid_ratio", 0.0)))
            empty_retrieval_values.append(float(stats.get("empty_retrieval_ratio", 0.0)))

        mean_stats = {}
        for key, values in stats_accumulator.items():
            finite_values = [value for value in values if math.isfinite(value)]
            mean_stats[key] = (
                float(sum(finite_values) / len(finite_values))
                if finite_values else float("nan")
            )
        if place_node_counts:
            mean_stats.update(
                {
                    "avg_place_nodes": float(sum(place_node_counts) / len(place_node_counts)),
                    "max_place_nodes_used": float(max(place_node_counts)),
                    "min_place_nodes_used": float(min(place_node_counts)),
                    "global_retrieved_nodes": float(sum(global_retrieved_counts) / len(global_retrieved_counts)),
                    "local_retrieved_nodes": float(sum(local_retrieved_counts) / len(local_retrieved_counts)),
                    "active_node_valid_ratio": float(sum(active_valid_values) / len(active_valid_values)),
                    "empty_retrieval_ratio": float(sum(empty_retrieval_values) / len(empty_retrieval_values)),
                }
            )
        return {
            "node_features_padded": node_features_padded,
            "global_retrieved_tokens_padded": node_features_padded,
            "node_positions_padded": node_positions_padded,
            "node_patch_padded": node_patch_padded,
            "node_padding_mask": node_padding_mask,
            "local_subgraph_tokens_padded": local_subgraph_tokens_padded,
            "local_token_padding_mask": local_token_padding_mask,
            "cell_to_node_map": cell_to_node_map,
            "neighbor_index_padded": neighbor_index_padded,
            "neighbor_mask_padded": neighbor_mask_padded,
            "local_context": local_context,
            "local_patch_context": local_patch_context,
            "stats": mean_stats,
        }
