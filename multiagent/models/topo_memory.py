import math

import torch
from torch import nn
from torch.nn import functional as F


def safe_cosine_similarity(a, b, dim=-1, eps=1e-6):
    a_norm = a / a.norm(dim=dim, keepdim=True).clamp_min(eps)
    b_norm = b / b.norm(dim=dim, keepdim=True).clamp_min(eps)
    return (a_norm * b_norm).sum(dim=dim)


def flattened_cell_id_to_xy(cell_ids, cell_positions):
    if cell_ids.numel() == 0:
        return cell_positions.new_zeros((0, cell_positions.shape[-1]))
    clamped_ids = cell_ids.long().clamp_(0, cell_positions.shape[0] - 1)
    return cell_positions.index_select(0, clamped_ids)


def bearing_change(prev_xy, curr_xy, next_xy, eps=1e-6):
    v1 = curr_xy - prev_xy
    v2 = next_xy - curr_xy
    n1 = v1.norm(dim=-1).clamp_min(eps)
    n2 = v2.norm(dim=-1).clamp_min(eps)
    cos_theta = (v1 * v2).sum(dim=-1) / (n1 * n2)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def masked_mean(x, mask, dim, keepdim=False):
    weights = (~mask).to(x.dtype)
    while weights.dim() < x.dim():
        weights = weights.unsqueeze(-1)
    total = (x * weights).sum(dim=dim, keepdim=keepdim)
    denom = weights.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return total / denom


class TopoGraphLayer(nn.Module):
    def __init__(self, hidden_size, edge_dim):
        super().__init__()
        self.self_proj = nn.Linear(hidden_size, hidden_size)
        self.neighbor_proj = nn.Linear(hidden_size, hidden_size)
        self.edge_proj = nn.Linear(edge_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, node_features, adjacency, edge_attr):
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_features = torch.matmul(adjacency, node_features) / degree
        edge_context = (adjacency.unsqueeze(-1) * edge_attr).sum(dim=-2) / degree
        update = self.self_proj(node_features)
        update = update + self.neighbor_proj(neighbor_features)
        update = update + self.edge_proj(edge_context)
        update = F.relu(self.out_proj(update))
        return self.norm(node_features + update)


class TopoGraphEncoder(nn.Module):
    def __init__(self, hidden_size, edge_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            TopoGraphLayer(hidden_size, edge_dim) for _ in range(max(int(num_layers), 0))
        )

    def forward(self, node_features, adjacency, edge_attr):
        output = node_features
        for layer in self.layers:
            output = layer(output, adjacency, edge_attr)
        return output


class TopoMemoryBuilder(nn.Module):
    EDGE_DIM = 9

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.graph_encoder = TopoGraphEncoder(
            args.demb,
            self.EDGE_DIM,
            args.topo_message_passing_layers,
        )
        self._eval_debug_batches = 0

    def _update_node(self, node, xy_t, weighted_feat, time_t, decay_weight, goal_relevance):
        momentum = float(self.args.topo_update_momentum)
        inv_momentum = 1.0 - momentum
        center_momentum = max(momentum, 0.9)
        center_inv_momentum = 1.0 - center_momentum
        node["summary_feat"] = momentum * node["summary_feat"] + inv_momentum * weighted_feat
        node["patch_feat"] = momentum * node["patch_feat"] + inv_momentum * weighted_feat
        node["semantic_feat"] = momentum * node["semantic_feat"] + inv_momentum * weighted_feat
        # Keep the spatial anchor stable: continuous xy updates node centers with a smaller EMA step.
        node["center_xy"] = center_momentum * node["center_xy"] + center_inv_momentum * xy_t
        node["visit_count"] += 1.0
        node["t_last"] = time_t
        node["importance"] += float(decay_weight.item()) + float(goal_relevance.item())

    def _create_node(self, nodes, xy_t, weighted_feat, time_t, decay_weight, goal_relevance):
        nodes.append(
            {
                "center_xy": xy_t.clone(),
                "summary_feat": weighted_feat.clone(),
                "patch_feat": weighted_feat.clone(),
                "semantic_feat": weighted_feat.clone(),
                "visit_count": 1.0,
                "t_first": time_t,
                "t_last": time_t,
                "importance": float(decay_weight.item()) + float(goal_relevance.item()),
            }
        )
        return len(nodes) - 1

    def _remap_visit_sequence(self, visit_sequence, remove_idx, keep_idx):
        remapped = []
        for visit_idx in visit_sequence:
            if visit_idx == remove_idx:
                remapped.append(keep_idx if keep_idx < remove_idx else keep_idx - 1)
            elif visit_idx > remove_idx:
                remapped.append(visit_idx - 1)
            else:
                remapped.append(visit_idx)
        visit_sequence[:] = remapped

    def _merge_duplicate_nodes(self, nodes, visit_sequence, duplicate_eps):
        if len(nodes) < 2:
            return False
        centers = torch.stack([node["center_xy"] for node in nodes], dim=0)
        dist = torch.cdist(centers, centers)
        dist.fill_diagonal_(float("inf"))
        min_dist, flat_idx = dist.view(-1).min(dim=0)
        if float(min_dist.item()) > duplicate_eps:
            return False
        node_count = dist.shape[0]
        remove_idx = int(flat_idx.item() // node_count)
        keep_idx = int(flat_idx.item() % node_count)
        if nodes[keep_idx]["visit_count"] < nodes[remove_idx]["visit_count"]:
            keep_idx, remove_idx = remove_idx, keep_idx
        self._merge_nodes(nodes, remove_idx, keep_idx)
        self._remap_visit_sequence(visit_sequence, remove_idx, keep_idx)
        return True

    def _edge_features(self, src_xy, dst_xy, delta_t, edge_type, device, dtype):
        delta = dst_xy - src_xy
        distance = delta.norm().clamp_min(1e-6)
        bearing_sin = delta[1] / distance
        bearing_cos = delta[0] / distance
        type_onehot = torch.zeros(3, device=device, dtype=dtype)
        type_onehot[edge_type] = 1.0
        edge_features = torch.cat(
            (
                delta,
                distance.view(1),
                bearing_sin.view(1),
                bearing_cos.view(1),
                delta_t.view(1),
                type_onehot,
            ),
            dim=0,
        )
        assert edge_features.shape[0] == self.EDGE_DIM, (
            f"TopoMemoryBuilder edge feature dim mismatch: "
            f"expected {self.EDGE_DIM}, got {edge_features.shape[0]}"
        )
        return edge_features

    def _merge_nodes(self, nodes, remove_idx, keep_idx):
        if remove_idx == keep_idx:
            return
        remove_node = nodes[remove_idx]
        keep_node = nodes[keep_idx]
        keep_visits = keep_node["visit_count"]
        remove_visits = remove_node["visit_count"]
        total_visits = keep_visits + remove_visits
        if total_visits <= 0:
            total_visits = 1.0
        keep_node["center_xy"] = (
            keep_node["center_xy"] * keep_visits + remove_node["center_xy"] * remove_visits
        ) / total_visits
        keep_node["summary_feat"] = (
            keep_node["summary_feat"] * keep_visits + remove_node["summary_feat"] * remove_visits
        ) / total_visits
        keep_node["patch_feat"] = (
            keep_node["patch_feat"] * keep_visits + remove_node["patch_feat"] * remove_visits
        ) / total_visits
        keep_node["semantic_feat"] = (
            keep_node["semantic_feat"] * keep_visits + remove_node["semantic_feat"] * remove_visits
        ) / total_visits
        keep_node["visit_count"] = total_visits
        keep_node["t_first"] = min(keep_node["t_first"], remove_node["t_first"])
        keep_node["t_last"] = max(keep_node["t_last"], remove_node["t_last"])
        keep_node["importance"] = keep_node["importance"] + remove_node["importance"]
        del nodes[remove_idx]

    def _enforce_max_nodes(self, nodes, visit_sequence):
        while len(nodes) > self.args.topo_max_nodes:
            centers = torch.stack([node["center_xy"] for node in nodes], dim=0)
            summary = torch.stack([node["summary_feat"] for node in nodes], dim=0)
            dist = torch.cdist(centers, centers)
            dist.fill_diagonal_(float("inf"))
            similarity = safe_cosine_similarity(summary.unsqueeze(1), summary.unsqueeze(0), dim=-1)
            similarity = similarity.clamp_min(0.0)
            similarity.fill_diagonal_(0.0)
            redundancy = similarity / (1.0 + dist.clamp_min(1e-2))
            importance = torch.tensor(
                [node["importance"] + node["visit_count"] for node in nodes],
                device=centers.device,
                dtype=centers.dtype,
            )
            remove_idx = int(torch.argmin(importance).item())
            keep_idx = int(torch.argmax(redundancy[remove_idx]).item())
            if keep_idx == remove_idx:
                keep_idx = int(torch.argmin(dist[remove_idx]).item())
            self._merge_nodes(nodes, remove_idx, keep_idx)
            self._remap_visit_sequence(visit_sequence, remove_idx, keep_idx)

    def _build_edges(self, centers, features, visit_sequence, visit_times):
        device = centers.device
        dtype = centers.dtype
        num_nodes = centers.shape[0]
        adjacency = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
        edge_attr = torch.zeros(num_nodes, num_nodes, self.EDGE_DIM, device=device, dtype=dtype)

        temporal_edges = 0
        spatial_edges = 0
        semantic_edges = 0

        for idx in range(1, len(visit_sequence)):
            src = visit_sequence[idx - 1]
            dst = visit_sequence[idx]
            if src == dst:
                continue
            delta_t = torch.tensor(
                float(max(visit_times[idx] - visit_times[idx - 1], 0.0)),
                device=device,
                dtype=dtype,
            )
            adjacency[src, dst] = adjacency[src, dst] + 1.0
            edge_attr[src, dst] = edge_attr[src, dst] + self._edge_features(
                centers[src], centers[dst], delta_t, 0, device, dtype
            )
            temporal_edges += 1

        if num_nodes > 1:
            dist = torch.cdist(centers, centers)
            similarity = safe_cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
            for node_idx in range(num_nodes):
                neighbor_order = torch.argsort(dist[node_idx])
                knn_count = 0
                for neighbor_idx in neighbor_order.tolist():
                    if neighbor_idx == node_idx:
                        continue
                    if dist[node_idx, neighbor_idx] <= self.args.topo_spatial_edge_radius:
                        delta_t = torch.tensor(0.0, device=device, dtype=dtype)
                        adjacency[node_idx, neighbor_idx] = 1.0
                        adjacency[neighbor_idx, node_idx] = 1.0
                        edge_attr[node_idx, neighbor_idx] = self._edge_features(
                            centers[node_idx], centers[neighbor_idx], delta_t, 1, device, dtype
                        )
                        edge_attr[neighbor_idx, node_idx] = self._edge_features(
                            centers[neighbor_idx], centers[node_idx], delta_t, 1, device, dtype
                        )
                        spatial_edges += 1
                    elif knn_count < self.args.topo_knn:
                        delta_t = torch.tensor(0.0, device=device, dtype=dtype)
                        adjacency[node_idx, neighbor_idx] = 1.0
                        adjacency[neighbor_idx, node_idx] = 1.0
                        edge_attr[node_idx, neighbor_idx] = self._edge_features(
                            centers[node_idx], centers[neighbor_idx], delta_t, 1, device, dtype
                        )
                        edge_attr[neighbor_idx, node_idx] = self._edge_features(
                            centers[neighbor_idx], centers[node_idx], delta_t, 1, device, dtype
                        )
                        knn_count += 1
                        spatial_edges += 1

                semantic_neighbors = torch.nonzero(
                    (similarity[node_idx] >= self.args.topo_semantic_edge_threshold)
                    & (torch.arange(num_nodes, device=device) != node_idx),
                    as_tuple=False,
                ).flatten()
                for neighbor_idx in semantic_neighbors.tolist():
                    delta_t = torch.tensor(0.0, device=device, dtype=dtype)
                    adjacency[node_idx, neighbor_idx] = 1.0
                    adjacency[neighbor_idx, node_idx] = 1.0
                    edge_attr[node_idx, neighbor_idx] = self._edge_features(
                        centers[node_idx], centers[neighbor_idx], delta_t, 2, device, dtype
                    )
                    edge_attr[neighbor_idx, node_idx] = self._edge_features(
                        centers[neighbor_idx], centers[node_idx], delta_t, 2, device, dtype
                    )
                    semantic_edges += 1

        adjacency.fill_diagonal_(1.0)
        return adjacency, edge_attr, temporal_edges, spatial_edges, semantic_edges

    def _compute_neighbor_index(self, adjacency, max_neighbors):
        num_nodes = adjacency.shape[0]
        if num_nodes == 0:
            return adjacency.new_zeros((0, max_neighbors), dtype=torch.long), adjacency.new_ones(
                (0, max_neighbors), dtype=torch.bool
            )
        adjacency_wo_self = adjacency.clone()
        adjacency_wo_self.fill_diagonal_(0.0)
        neighbor_index = torch.zeros(num_nodes, max_neighbors, device=adjacency.device, dtype=torch.long)
        neighbor_mask = torch.ones(num_nodes, max_neighbors, device=adjacency.device, dtype=torch.bool)
        for node_idx in range(num_nodes):
            scores, indices = torch.topk(
                adjacency_wo_self[node_idx],
                k=min(max_neighbors, num_nodes),
                largest=True,
            )
            valid = scores > 0
            if valid.any():
                count = int(valid.sum().item())
                neighbor_index[node_idx, :count] = indices[:count]
                neighbor_mask[node_idx, :count] = False
        return neighbor_index, neighbor_mask

    def build_single(
        self,
        history_features,
        history_cell_ids,
        history_times,
        history_xy,
        base_positions,
        lang_goal_embed,
        current_grid,
        fallback_feature,
        time_decay_rate,
        debug_eval=False,
        debug_batch_idx=0,
    ):
        device = history_features.device if history_features.numel() > 0 else fallback_feature.device
        dtype = fallback_feature.dtype
        current_grid = int(current_grid)
        cell_count = base_positions.shape[0]

        nodes = []
        visit_sequence = []
        visit_times = []
        created_nodes = 0.0
        updated_nodes = 0.0
        goal_relevances = []
        novelties = []
        nodes_before_merge = 0.0
        duplicate_eps = min(float(self.args.topo_merge_radius) * 0.25, 1e-3)

        prev_prev_xy = None
        prev_xy = None

        for step_idx in range(history_features.shape[0]):
            feat_t = history_features[step_idx]
            cell_id = int(history_cell_ids[step_idx].item())
            fallback_xy_t = base_positions[cell_id]
            # Prefer continuous rollout xy for topo memory. Cell centers remain only as a compatibility fallback.
            if history_xy.numel() > 0 and history_xy.shape[0] > step_idx:
                xy_candidate = history_xy[step_idx]
                if torch.isfinite(xy_candidate).all():
                    xy_t = xy_candidate.to(device=device, dtype=base_positions.dtype)
                else:
                    xy_t = fallback_xy_t
            else:
                xy_t = fallback_xy_t
            time_t = float(history_times[step_idx].item()) if history_times.numel() > step_idx else float(step_idx)
            time_gap = history_times[-1] - history_times[step_idx] if history_times.numel() > 0 else feat_t.new_tensor(0.0)
            decay_weight = torch.exp(-time_decay_rate * time_gap.clamp_min(0.0))
            weighted_feat = feat_t * decay_weight

            goal_relevance = safe_cosine_similarity(
                weighted_feat.unsqueeze(0), lang_goal_embed.unsqueeze(0), dim=-1
            ).squeeze(0)
            goal_relevances.append(float(goal_relevance.item()))

            turning_event = False
            if prev_prev_xy is not None and prev_xy is not None:
                turn_deg = bearing_change(prev_prev_xy.unsqueeze(0), prev_xy.unsqueeze(0), xy_t.unsqueeze(0))[0]
                turning_event = bool(turn_deg.item() >= self.args.topo_turn_threshold_deg)

            if nodes:
                existing_centers = torch.stack([node["center_xy"] for node in nodes], dim=0)
                existing_summary = torch.stack([node["summary_feat"] for node in nodes], dim=0)
                distance = torch.cdist(xy_t.unsqueeze(0), existing_centers).squeeze(0)
                nearest_idx = int(torch.argmin(distance).item())
                nearest_dist = distance[nearest_idx]
                similarity_to_nodes = safe_cosine_similarity(
                    weighted_feat.unsqueeze(0), existing_summary, dim=-1
                )
                nearest_similarity = similarity_to_nodes[nearest_idx]
                novelty = 1.0 - similarity_to_nodes.max()
            else:
                nearest_idx = -1
                nearest_dist = fallback_feature.new_tensor(float("inf"))
                nearest_similarity = fallback_feature.new_tensor(0.0)
                novelty = fallback_feature.new_tensor(1.0)

            novelties.append(float(novelty.item()))

            update_existing = False
            if len(nodes) == 0:
                create_new = True
            elif nearest_dist.item() <= self.args.topo_merge_radius:
                # Spatial anchoring wins: repeated same-cell observations update by default.
                create_new = False
                update_existing = True
            elif nearest_dist.item() > self.args.topo_create_radius:
                create_new = True
            else:
                create_new = bool(
                    turning_event
                    and nearest_dist.item() > duplicate_eps
                    and goal_relevance.item() >= self.args.topo_goal_rel_threshold
                    and novelty.item() > self.args.topo_novelty_threshold
                    and nearest_similarity.item() < self.args.topo_merge_sim_threshold
                )
                update_existing = not create_new

            if create_new:
                node_idx = self._create_node(
                    nodes, xy_t, weighted_feat, time_t, decay_weight, goal_relevance
                )
                created_nodes += 1.0
                nodes_before_merge = max(nodes_before_merge, float(len(nodes)))
            else:
                node = nodes[nearest_idx]
                self._update_node(node, xy_t, weighted_feat, time_t, decay_weight, goal_relevance)
                node_idx = nearest_idx
                update_existing = True
                updated_nodes += 1.0

            visit_sequence.append(node_idx)
            visit_times.append(time_t)
            while self._merge_duplicate_nodes(nodes, visit_sequence, duplicate_eps):
                pass
            self._enforce_max_nodes(nodes, visit_sequence)
            prev_prev_xy = prev_xy
            prev_xy = xy_t
            if debug_eval and step_idx < 8:
                print(
                    '[TopoEvalStep] batch={} step_idx={} cell_id={} xy_t={} create_new={} update_existing={} '
                    'nearest_dist={:.4f} nearest_similarity={:.4f} novelty={:.4f} '
                    'goal_relevance={:.4f} nodes_now={}'.format(
                        debug_batch_idx,
                        step_idx,
                        cell_id,
                        xy_t.detach().cpu().tolist(),
                        bool(create_new),
                        bool(update_existing),
                        float(nearest_dist.item()),
                        float(nearest_similarity.item()),
                        float(novelty.item()),
                        float(goal_relevance.item()),
                        len(nodes),
                    )
                )

        if not nodes:
            fallback_xy = base_positions[current_grid].clone()
            fallback_node = {
                "center_xy": fallback_xy,
                "summary_feat": fallback_feature.clone(),
                "patch_feat": fallback_feature.clone(),
                "semantic_feat": fallback_feature.clone(),
                "visit_count": 1.0,
                "t_first": 0.0,
                "t_last": 0.0,
                "importance": 1.0,
            }
            nodes.append(fallback_node)
            visit_sequence.append(0)
            visit_times.append(0.0)
            created_nodes = max(created_nodes, 1.0)
            nodes_before_merge = max(nodes_before_merge, 1.0)

        node_features = torch.stack([node["summary_feat"] for node in nodes], dim=0).to(dtype)
        node_positions = torch.stack([node["center_xy"] for node in nodes], dim=0).to(dtype)
        node_patch = torch.stack([node["patch_feat"] for node in nodes], dim=0).to(dtype)
        node_visits = torch.tensor(
            [node["visit_count"] for node in nodes], device=device, dtype=dtype
        )

        adjacency, edge_attr, temporal_edges, spatial_edges, semantic_edges = self._build_edges(
            node_positions, node_features, visit_sequence, visit_times
        )
        if self.args.topo_use_graph_encoder and len(self.graph_encoder.layers) > 0:
            node_features = self.graph_encoder(node_features, adjacency, edge_attr)

        distances = torch.cdist(base_positions.to(dtype), node_positions)
        cell_to_node_map = torch.argmin(distances, dim=1).long()
        neighbor_index, neighbor_mask = self._compute_neighbor_index(
            adjacency,
            max(int(self.args.topo_patch_topk), 1),
        )

        stats = {
            "nodes_before_merge": nodes_before_merge or float(node_features.shape[0]),
            "nodes_after_merge": float(node_features.shape[0]),
            "created_nodes": created_nodes,
            "updated_nodes": updated_nodes,
            "temporal_edges": float(temporal_edges),
            "spatial_edges": float(spatial_edges),
            "semantic_edges": float(semantic_edges),
            "avg_goal_relevance": float(sum(goal_relevances) / max(len(goal_relevances), 1)),
            "avg_novelty": float(sum(novelties) / max(len(novelties), 1)),
            "avg_node_visits": float(node_visits.mean().item()),
        }

        return {
            "node_features": node_features,
            "node_positions": node_positions,
            "node_patch": node_patch,
            "cell_to_node_map": cell_to_node_map,
            "neighbor_index": neighbor_index,
            "neighbor_mask": neighbor_mask,
            "stats": stats,
        }

    def forward(
        self,
        history_features,
        history_cell_ids,
        history_times,
        history_xy,
        base_positions,
        lang_goal_embed,
        current_grids,
        fallback_features,
        time_decay_rate,
    ):
        batch_outputs = []
        max_nodes = 1
        max_neighbors = max(int(self.args.topo_patch_topk), 1)
        debug_eval = (not self.training and self._eval_debug_batches < 3)
        debug_batch_idx = self._eval_debug_batches
        for batch_idx in range(history_features.shape[0]):
            output = self.build_single(
                history_features[batch_idx],
                history_cell_ids[batch_idx].long(),
                history_times[batch_idx],
                history_xy[batch_idx],
                base_positions[batch_idx],
                lang_goal_embed[batch_idx],
                current_grids[batch_idx].item(),
                fallback_features[batch_idx],
                time_decay_rate,
                debug_eval=debug_eval,
                debug_batch_idx=debug_batch_idx,
            )
            batch_outputs.append(output)
            max_nodes = max(max_nodes, output["node_features"].shape[0])
        if debug_eval:
            self._eval_debug_batches += 1

        node_features_padded = history_features.new_zeros(
            history_features.shape[0], max_nodes, history_features.shape[-1]
        )
        node_positions_padded = base_positions.new_zeros(history_features.shape[0], max_nodes, 2)
        node_patch_padded = history_features.new_zeros(
            history_features.shape[0], max_nodes, history_features.shape[-1]
        )
        node_padding_mask = torch.ones(
            history_features.shape[0], max_nodes, device=history_features.device, dtype=torch.bool
        )
        cell_to_node_map = torch.zeros(
            history_features.shape[0], base_positions.shape[1], device=history_features.device, dtype=torch.long
        )
        neighbor_index_padded = torch.zeros(
            history_features.shape[0], max_nodes, max_neighbors, device=history_features.device, dtype=torch.long
        )
        neighbor_mask_padded = torch.ones(
            history_features.shape[0], max_nodes, max_neighbors, device=history_features.device, dtype=torch.bool
        )

        stats_accumulator = {}
        for batch_idx, output in enumerate(batch_outputs):
            count = output["node_features"].shape[0]
            node_features_padded[batch_idx, :count] = output["node_features"]
            node_positions_padded[batch_idx, :count] = output["node_positions"]
            node_patch_padded[batch_idx, :count] = output["node_patch"]
            node_padding_mask[batch_idx, :count] = False
            cell_to_node_map[batch_idx] = output["cell_to_node_map"]
            neighbor_index_padded[batch_idx, :count, : output["neighbor_index"].shape[1]] = output["neighbor_index"]
            neighbor_mask_padded[batch_idx, :count, : output["neighbor_mask"].shape[1]] = output["neighbor_mask"]
            for key, value in output["stats"].items():
                stats_accumulator.setdefault(key, []).append(value)

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
            "stats": mean_stats,
        }
