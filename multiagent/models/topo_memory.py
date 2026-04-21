"""Persistent instruction-conditioned topological memory for HETT."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    attached_place_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0
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
        self.event_meta = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.place_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.landmark_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.event_type = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.event_type_embed = nn.Embedding(len(EVENT_TYPE_TO_ID), hidden_size)

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
                float(place.goal_relevance),
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode a landmark node token."""

        device, dtype = self._resolve_module_spec(self.landmark_meta, device=device, dtype=dtype)
        semantic_embedding = landmark.semantic_embedding.to(device=device, dtype=dtype)
        _, stats = self._landmark_stats_tensor(landmark, device=device, dtype=dtype)
        return semantic_embedding + self.landmark_meta(stats) + self.landmark_type.to(device=device, dtype=dtype)

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
        self._step_updated_nodes = 0
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
                novelty_score=float(novelty.item()),
            )
        )
        self._step_new_nodes += 1
        return place_id

    def _update_place_node(
        self,
        place_id: int,
        xy_t: torch.Tensor,
        heading_vec: torch.Tensor,
        feature_t: torch.Tensor,
        step_id: int,
        goal_relevance: torch.Tensor,
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
        place.novelty_score = momentum * place.novelty_score + (1.0 - momentum) * float(novelty.item())
        for name in landmark_names:
            if name and name not in place.observed_landmarks:
                place.observed_landmarks.append(name)
        place.recent_patch_bank.append(feature_t.clone())
        if len(place.recent_patch_bank) > int(self.args.patch_bank_size):
            place.recent_patch_bank.pop(0)
        self._step_updated_nodes += 1
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

    def _update_landmarks(self, place_id: int, step_id: int, landmark_info: Sequence[dict]) -> None:
        """Update persistent landmark nodes and semantic attachments."""

        if not bool(self.args.use_landmark_nodes):
            return
        for item in landmark_info:
            name = item.get("name", "")
            if not name:
                continue
            centroid = item.get("centroid")
            polygon = item.get("polygon")
            if centroid is None:
                centroid_tensor = self._place_by_id(place_id).center_xy.clone()
            elif isinstance(centroid, torch.Tensor):
                centroid_tensor = centroid.to(device=self.instruction_feat.device, dtype=self.instruction_feat.dtype)
            else:
                centroid_tensor = torch.tensor(
                    centroid,
                    device=self.instruction_feat.device,
                    dtype=self.instruction_feat.dtype,
                )

            if name in self._landmark_index:
                landmark = self.landmark_nodes[self._landmark_index[name]]
                landmark.last_seen_step = step_id
                landmark.confidence = min(1.0, landmark.confidence + 0.1)
                landmark.center_xy = centroid_tensor
                if place_id not in landmark.attached_place_ids:
                    landmark.attached_place_ids.append(place_id)
                    self._step_landmark_attachments += 1
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
                        attached_place_ids=[place_id],
                        confidence=0.5,
                        last_seen_step=step_id,
                        center_xy=centroid_tensor,
                    )
                )
                self._landmark_index[name] = landmark_id
                self._step_landmark_attachments += 1
                self._create_event("first_landmark", place_id, step_id, 0.5)
            self.semantic_edges.append((place_id, self._landmark_index[name]))

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
            importance = torch.tensor(
                [node.visit_count + node.goal_relevance for node in self.place_nodes],
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
                for place_id in landmark.attached_place_ids:
                    mapped = keep_id if place_id == remove_id else place_id
                    if mapped not in updated_ids:
                        updated_ids.append(mapped)
                landmark.attached_place_ids = updated_ids
            for event in self.event_nodes:
                if event.attached_place_id == remove_id:
                    event.attached_place_id = keep_id
            self._remap_edges(remove_id, keep_id)
            if self._active_place_id == remove_id:
                self._active_place_id = keep_id
            del self.place_nodes[remove_slot]
            self._step_merged_nodes += 1

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
        time_decay_rate: torch.Tensor,
    ) -> Dict[str, float]:
        """Incrementally update the persistent graph with one new observation."""

        self._require_started()
        self._step_new_nodes = 0
        self._step_merged_nodes = 0
        self._step_landmark_attachments = 0
        self._step_updated_nodes = 0

        base_xy = self.base_positions[int(cell_id)]
        if xy is not None and torch.isfinite(xy).all():
            xy_t = xy.to(device=self.base_positions.device, dtype=self.base_positions.dtype)
        else:
            xy_t = base_xy

        feature_t = observation_feature.to(self.instruction_feat.dtype)
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

        if not association_scores:
            selected_place_id = self._create_place_node(
                xy_t,
                heading_vec,
                feature_t,
                step_id,
                goal_relevance,
                novelty,
                landmark_names,
            )
        else:
            best_place_id, best_score_tensor = max(association_scores, key=lambda item: float(item[1].item()))
            create_new = any(proposal_flags.values()) or float(best_score_tensor.item()) < float(self.args.topo_merge_sim_threshold)
            if create_new:
                selected_place_id = self._create_place_node(
                    xy_t,
                    heading_vec,
                    feature_t,
                    step_id,
                    goal_relevance,
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
                    novelty,
                    landmark_names,
                )

        if self._active_place_id is not None and self._active_place_id != selected_place_id:
            self.temporal_edges.append((self._active_place_id, selected_place_id))
            self._create_event("branch", selected_place_id, step_id, float(goal_relevance.item()))
        self._active_place_id = selected_place_id

        if heading_change_deg >= float(self.args.turn_event_threshold_deg):
            self._create_event("turn", selected_place_id, step_id, heading_change_deg / 180.0)
        if self._prev_goal_relevance is not None:
            jump = abs(float(goal_relevance.item()) - self._prev_goal_relevance)
            if jump >= float(self.args.relevance_jump_threshold):
                self._create_event("relevance_jump", selected_place_id, step_id, jump)
        self._prev_goal_relevance = float(goal_relevance.item())

        self._update_landmarks(selected_place_id, step_id, landmark_info)
        self._enforce_max_place_nodes()
        self._refresh_spatial_edges()
        self._prev_prev_xy = self._prev_xy
        self._prev_xy = xy_t

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
        time_decay_rate: torch.Tensor,
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

    def retrieve_global_graph_tokens(
        self,
        instruction_feat: torch.Tensor,
        k: int,
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
            place_token = self._normalize_feature_matrix(
                self._to_device(self.fallback_feature, device, hidden_dtype),
                name="retrieve_global_graph_tokens.place_tokens[fallback]",
            )
            place_position = self._to_device(self.base_positions[:1].clone(), device, self.base_positions.dtype)
            place_ids = torch.zeros(1, device=device, dtype=torch.long)
            return {
                "place_tokens": place_token,
                "place_positions": place_position,
                "place_ids": place_ids,
                "landmark_tokens": self._normalize_feature_matrix(
                    place_token.new_zeros((0, place_token.shape[-1])),
                    name="retrieve_global_graph_tokens.landmark_tokens[empty]",
                ),
                "event_tokens": self._normalize_feature_matrix(
                    place_token.new_zeros((0, place_token.shape[-1])),
                    name="retrieve_global_graph_tokens.event_tokens[empty]",
                ),
            }

        ranked_ids = []
        for place in self.place_nodes:
            visual_score = safe_cosine_similarity(
                place.visual_prototype(device=device, dtype=hidden_dtype).unsqueeze(0),
                instruction_feat.unsqueeze(0),
            ).item()
            score = 0.7 * place.goal_relevance + 0.2 * visual_score + 0.1 * place.novelty_score
            ranked_ids.append((place.node_id, score))
        ranked_ids.sort(key=lambda item: item[1], reverse=True)
        selected_ids = [node_id for node_id, _ in ranked_ids[: max(int(k), 1)]]
        selected_places = [self._place_by_id(node_id) for node_id in selected_ids]

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

        landmark_tokens = [
            self.token_encoder.encode_landmark(landmark, device=device, dtype=hidden_dtype)
            for landmark in self.landmark_nodes
            if selected_set.intersection(landmark.attached_place_ids)
        ]
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
            "landmark_tokens": self._normalize_feature_matrix(
                torch.stack(landmark_tokens, dim=0) if landmark_tokens else place_tokens.new_zeros((0, place_tokens.shape[-1])),
                name="retrieve_global_graph_tokens.landmark_tokens",
            ),
            "event_tokens": self._normalize_feature_matrix(
                torch.stack(event_tokens, dim=0) if event_tokens else place_tokens.new_zeros((0, place_tokens.shape[-1])),
                name="retrieve_global_graph_tokens.event_tokens",
            ),
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
            }

        if active_node_id is None:
            active_node_id = self.place_nodes[0].node_id

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

        landmark_tokens = [
            self.token_encoder.encode_landmark(landmark, device=device, dtype=hidden_dtype)
            for landmark in self.landmark_nodes
            if set(landmark.attached_place_ids).intersection(ordered_ids)
        ]
        ref_step = max(place.last_seen_step for place in self.place_nodes)
        event_tokens = [
            self.token_encoder.encode_event(event, ref_step, hidden_dtype, device)
            for event in self.event_nodes
            if event.attached_place_id in connected_ids
        ]
        local_tokens = [place_tokens]
        if landmark_tokens:
            local_tokens.append(
                self._normalize_feature_matrix(
                    torch.stack(landmark_tokens, dim=0),
                    name="retrieve_local_subgraph_tokens.landmark_tokens",
                )
            )
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
                "stats": {
                    "nodes_after_merge": 0.0,
                    "created_nodes": 0.0,
                    "updated_nodes": 0.0,
                    "merged_nodes": 0.0,
                    "landmark_nodes": 0.0,
                    "event_nodes": 0.0,
                    "temporal_edges": 0.0,
                    "spatial_edges": 0.0,
                    "semantic_edges": 0.0,
                    "reobservation_edges": 0.0,
                    "active_place_id": -1.0,
                },
            }

        global_out = self.retrieve_global_graph_tokens(
            self.instruction_feat,
            int(self.args.global_retrieve_k),
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
            "nodes_after_merge": float(len(self.place_nodes)),
            "created_nodes": float(self._step_new_nodes),
            "updated_nodes": float(self._step_updated_nodes),
            "merged_nodes": float(self._step_merged_nodes),
            "landmark_nodes": float(len(self.landmark_nodes)),
            "event_nodes": float(len(self.event_nodes)),
            "temporal_edges": float(len(self.temporal_edges)),
            "spatial_edges": float(len(self.spatial_edges)),
            "semantic_edges": float(len(self.semantic_edges)),
            "reobservation_edges": float(len(self.reobservation_edges)),
            "active_place_id": float(self._active_place_id) if self._active_place_id is not None else -1.0,
        }
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
        time_decay_rate: torch.Tensor,
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
        time_decay_rate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Batch the rebuild-from-history fallback outputs."""

        outputs = []
        max_nodes = 1
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

        node_features_padded = history_features.new_zeros(history_features.shape[0], max_nodes, history_features.shape[-1])
        node_positions_padded = base_positions.new_zeros(history_features.shape[0], max_nodes, 2)
        node_patch_padded = history_features.new_zeros(history_features.shape[0], max_nodes, history_features.shape[-1])
        node_padding_mask = torch.ones(history_features.shape[0], max_nodes, device=history_features.device, dtype=torch.bool)
        cell_to_node_map = torch.zeros(history_features.shape[0], base_positions.shape[1], device=history_features.device, dtype=torch.long)
        neighbor_index_padded = torch.zeros(history_features.shape[0], max_nodes, max_neighbors, device=history_features.device, dtype=torch.long)
        neighbor_mask_padded = torch.ones(history_features.shape[0], max_nodes, max_neighbors, device=history_features.device, dtype=torch.bool)
        local_context = history_features.new_zeros(history_features.shape[0], history_features.shape[-1])
        local_patch_context = history_features.new_zeros(history_features.shape[0], history_features.shape[-1])
        stats_accumulator: Dict[str, List[float]] = {}

        for batch_idx, output in enumerate(outputs):
            count = int(output["node_features"].shape[0])
            node_features_padded[batch_idx, :count] = output["node_features"]
            node_positions_padded[batch_idx, :count] = output["node_positions"]
            node_patch_padded[batch_idx, :count] = output["node_patch"]
            node_padding_mask[batch_idx, :count] = False
            cell_to_node_map[batch_idx] = output["cell_to_node_map"]
            neighbor_index = output["neighbor_index"]
            neighbor_mask = output["neighbor_mask"]
            neighbor_index_padded[batch_idx, :count, : neighbor_index.shape[1]] = neighbor_index
            neighbor_mask_padded[batch_idx, :count, : neighbor_mask.shape[1]] = neighbor_mask
            local_context[batch_idx] = output["local_context"]
            local_patch_context[batch_idx] = output["local_patch_context"]
            for key, value in output["stats"].items():
                stats_accumulator.setdefault(key, []).append(float(value))

        mean_stats = {
            key: float(sum(values) / max(len(values), 1))
            for key, values in stats_accumulator.items()
        }
        return {
            "node_features_padded": node_features_padded,
            "node_positions_padded": node_positions_padded,
            "node_patch_padded": node_patch_padded,
            "node_padding_mask": node_padding_mask,
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
        time_decay_rate: torch.Tensor,
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
        max_neighbors = max(int(output["neighbor_index"].shape[1]) for output in outputs)
        batch_size = len(outputs)
        hidden_size = template_tensor.shape[-1]

        node_features_padded = template_tensor.new_zeros(batch_size, max_nodes, hidden_size)
        node_positions_padded = template_tensor.new_zeros(batch_size, max_nodes, 2)
        node_patch_padded = template_tensor.new_zeros(batch_size, max_nodes, hidden_size)
        node_padding_mask = torch.ones(batch_size, max_nodes, device=template_tensor.device, dtype=torch.bool)
        cell_to_node_map = torch.zeros(batch_size, self.graphs[0].base_positions.shape[0], device=template_tensor.device, dtype=torch.long)
        neighbor_index_padded = torch.zeros(batch_size, max_nodes, max_neighbors, device=template_tensor.device, dtype=torch.long)
        neighbor_mask_padded = torch.ones(batch_size, max_nodes, max_neighbors, device=template_tensor.device, dtype=torch.bool)
        local_context = template_tensor.new_zeros(batch_size, hidden_size)
        local_patch_context = template_tensor.new_zeros(batch_size, hidden_size)
        stats_accumulator: Dict[str, List[float]] = {}

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
            cell_to_node_map[env_idx] = output["cell_to_node_map"].to(template_tensor.device)
            neighbor_index = output["neighbor_index"].to(template_tensor.device)
            neighbor_mask = output["neighbor_mask"].to(template_tensor.device)
            neighbor_index_padded[env_idx, :count, : neighbor_index.shape[1]] = neighbor_index
            neighbor_mask_padded[env_idx, :count, : neighbor_mask.shape[1]] = neighbor_mask
            local_context[env_idx] = output["local_context"].to(template_tensor.device)
            local_patch_context[env_idx] = output["local_patch_context"].to(template_tensor.device)
            for key, value in output["stats"].items():
                stats_accumulator.setdefault(key, []).append(float(value))

        mean_stats = {
            key: float(sum(values) / max(len(values), 1))
            for key, values in stats_accumulator.items()
        }
        return {
            "node_features_padded": node_features_padded,
            "node_positions_padded": node_positions_padded,
            "node_patch_padded": node_patch_padded,
            "node_padding_mask": node_padding_mask,
            "cell_to_node_map": cell_to_node_map,
            "neighbor_index_padded": neighbor_index_padded,
            "neighbor_mask_padded": neighbor_mask_padded,
            "local_context": local_context,
            "local_patch_context": local_patch_context,
            "stats": mean_stats,
        }
