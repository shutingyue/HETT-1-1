"""Unified Memory Token Interface (UMTI-lite) for HETT.

Stage 1 intentionally keeps this adapter narrow: it wraps existing HETT grid
memory tokens, optional place-only topo tokens, and reserved future packs into
one masked memory-token interface.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import warnings

import torch
import torch.distributed as dist
from torch import nn


GRID_MEMORY_TYPE_ID = 0
TOPO_PLACE_TYPE_ID = 1
SEMANTIC_TYPE_ID = 2
AUXILIARY_TYPE_ID = 3


@dataclass
class MemoryPack:
    tokens: torch.Tensor
    mask: torch.Tensor
    positions: torch.Tensor
    type_ids: torch.Tensor
    scores: Optional[torch.Tensor] = None
    stats: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "tokens": self.tokens,
            "mask": self.mask,
            "positions": self.positions,
            "type_ids": self.type_ids,
            "scores": self.scores,
            "stats": self.stats or {},
        }


def _rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _stable_positions(batch_size: int, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if num_tokens <= 0:
        return torch.zeros(batch_size, 0, 2, device=device, dtype=dtype)
    side = int(num_tokens ** 0.5)
    if side * side == num_tokens:
        denom = max(side - 1, 1)
        rows = torch.arange(side, device=device, dtype=dtype)
        cols = torch.arange(side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        positions = torch.stack((yy.reshape(-1) / denom, xx.reshape(-1) / denom), dim=-1)
    else:
        x = torch.linspace(0.0, 1.0, steps=num_tokens, device=device, dtype=dtype)
        positions = torch.stack((x, torch.zeros_like(x)), dim=-1)
    return positions.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def build_grid_pack(historical_memory, positions=None, hidden_size=None) -> Dict[str, object]:
    return UMTIMemoryAdapter.build_grid_pack_static(historical_memory, positions, hidden_size).to_dict()


def build_topo_pack(topo_memory_state, batch=None, hidden_size=None) -> Dict[str, object]:
    return UMTIMemoryAdapter.build_topo_pack_static(topo_memory_state, batch, hidden_size).to_dict()


def build_semantic_pack_placeholder(batch=None, hidden_size=None, device=None, dtype=None) -> Dict[str, object]:
    return UMTIMemoryAdapter.build_semantic_pack_placeholder_static(
        batch=batch,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
    ).to_dict()


def fuse_memory_packs(packs: Iterable[Dict[str, object]]) -> Dict[str, object]:
    normalized = []
    for pack in packs:
        if pack is None:
            continue
        if isinstance(pack, MemoryPack):
            normalized.append(pack)
        else:
            normalized.append(MemoryPack(**pack))
    return UMTIMemoryAdapter.fuse_memory_packs_static(normalized).to_dict()


def debug_memory_stats(memory_pack: Dict[str, object]) -> Dict[str, object]:
    stats = dict(memory_pack.get("stats", {}))
    tokens = memory_pack["tokens"]
    mask = memory_pack["mask"]
    type_ids = memory_pack["type_ids"]
    stats.update(
        {
            "memory_tokens.shape": tuple(tokens.shape),
            "memory_mask.sum": int(mask.sum().item()),
            "num_grid_tokens": int(((type_ids == GRID_MEMORY_TYPE_ID) & mask).sum().item()),
            "num_topo_tokens": int(((type_ids == TOPO_PLACE_TYPE_ID) & mask).sum().item()),
            "num_semantic_tokens": int(((type_ids == SEMANTIC_TYPE_ID) & mask).sum().item()),
        }
    )
    return stats


class UMTIMemoryAdapter(nn.Module):
    def __init__(self, args, hidden_size: int):
        super().__init__()
        self.args = args
        self.hidden_size = int(hidden_size)
        self.num_memory_types = max(int(getattr(args, "num_memory_types", 4)), 4)
        self.use_memory_type_embedding = bool(getattr(args, "use_memory_type_embedding", False))
        self.use_topo_gate = bool(getattr(args, "use_topo_gate", True))
        self.debug_memory_tokens = bool(getattr(args, "debug_memory_tokens", False))
        self._debug_counter = 0
        self._debug_printed = set()
        self._semantic_warning_emitted = False

        self.position_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.memory_type_embedding = nn.Embedding(self.num_memory_types, self.hidden_size)
        self.topo_projection = None
        self.grid_projection = None
        self.topo_gate_logit = nn.Parameter(torch.tensor(-2.0))

    def _project_tokens_if_needed(self, tokens: torch.Tensor, attr_name: str) -> torch.Tensor:
        if tokens.shape[-1] == self.hidden_size:
            return tokens
        projection = getattr(self, attr_name)
        if projection is None or projection.in_features != tokens.shape[-1]:
            projection = nn.Linear(tokens.shape[-1], self.hidden_size).to(device=tokens.device, dtype=tokens.dtype)
            setattr(self, attr_name, projection)
        return projection(tokens)

    @staticmethod
    def _as_sequence_tokens(memory: torch.Tensor) -> torch.Tensor:
        if memory.dim() == 2:
            return memory.unsqueeze(1)
        if memory.dim() == 3:
            return memory
        if memory.dim() == 4:
            batch_size = memory.shape[0]
            return memory.reshape(batch_size, -1, memory.shape[-1])
        raise ValueError(f"UMTI expected memory rank 2, 3, or 4, got shape {tuple(memory.shape)}")

    @staticmethod
    def build_grid_pack_static(historical_memory, positions=None, hidden_size=None) -> MemoryPack:
        if not torch.is_tensor(historical_memory):
            raise TypeError("historical_memory must be a torch.Tensor")
        tokens = UMTIMemoryAdapter._as_sequence_tokens(historical_memory).to(torch.float32)
        batch_size, num_tokens, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        if positions is None:
            pos = _stable_positions(batch_size, num_tokens, device, dtype)
        else:
            pos = positions.to(device=device, dtype=dtype)
            if pos.dim() == 2:
                pos = pos.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            elif pos.dim() == 4:
                pos = pos.reshape(batch_size, -1, pos.shape[-1])
            if pos.shape[:2] != tokens.shape[:2] or pos.shape[-1] != 2:
                pos = _stable_positions(batch_size, num_tokens, device, dtype)
            else:
                pos = pos.clamp(0.0, 1.0)
        mask = torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)
        type_ids = torch.full((batch_size, num_tokens), GRID_MEMORY_TYPE_ID, device=device, dtype=torch.long)
        scores = torch.ones(batch_size, num_tokens, device=device, dtype=dtype)
        return MemoryPack(
            tokens=tokens,
            mask=mask,
            positions=pos,
            type_ids=type_ids,
            scores=scores,
            stats={
                "grid_tokens.shape": tuple(tokens.shape),
                "grid_mask.sum": int(mask.sum().item()),
                "num_grid_tokens": int(mask.sum().item()),
            },
        )

    @staticmethod
    def build_topo_pack_static(topo_memory_state, batch=None, hidden_size=None) -> MemoryPack:
        if topo_memory_state is None:
            if batch is None:
                raise ValueError("batch metadata is required when topo_memory_state is None")
            batch_size = int(batch["batch_size"])
            max_nodes = int(batch.get("topo_max_nodes", 0))
            hidden = int(hidden_size or batch["hidden_size"])
            device = batch["device"]
            dtype = batch.get("dtype", torch.float32)
            tokens = torch.zeros(batch_size, max_nodes, hidden, device=device, dtype=dtype)
            positions = torch.zeros(batch_size, max_nodes, 2, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.bool)
        else:
            source_tokens = topo_memory_state.get(
                "global_retrieved_tokens_padded",
                topo_memory_state.get("node_features_padded", None),
            )
            if source_tokens is None:
                raise KeyError("topo_memory_state must contain node_features_padded or global_retrieved_tokens_padded")
            tokens = source_tokens.to(torch.float32)
            batch_size = tokens.shape[0]
            if batch:
                max_nodes = int(batch.get("topo_max_nodes", tokens.shape[1]))
                max_nodes = min(max_nodes, int(batch.get("global_retrieve_k", max_nodes)))
            else:
                max_nodes = tokens.shape[1]
            tokens = tokens[:, :max_nodes]
            padding_mask = topo_memory_state.get("node_padding_mask", None)
            if padding_mask is None:
                mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
            else:
                mask = ~padding_mask[:, :max_nodes].to(device=tokens.device).bool()
            valid_place_mask = topo_memory_state.get("node_valid_place_mask", None)
            if valid_place_mask is not None:
                mask = mask & valid_place_mask[:, :max_nodes].to(device=tokens.device).bool()
            source_positions = topo_memory_state.get("node_positions_padded", None)
            if source_positions is None:
                positions = _stable_positions(batch_size, tokens.shape[1], tokens.device, tokens.dtype)
            else:
                positions = source_positions[:, :max_nodes].to(device=tokens.device, dtype=tokens.dtype).clamp(0.0, 1.0)
        if not bool(mask.any().item()):
            tokens = tokens[:, :0]
            positions = positions[:, :0]
            mask = mask[:, :0]
        type_ids = torch.full(tokens.shape[:2], TOPO_PLACE_TYPE_ID, device=tokens.device, dtype=torch.long)
        scores = mask.to(tokens.dtype)
        stats = {
            "topo_tokens.shape": tuple(tokens.shape),
            "topo_mask.sum": int(mask.sum().item()),
            "topo_mask_sum": int(mask.sum().item()),
            "num_topo_tokens": int(mask.sum().item()),
            "num_valid_topo_tokens_to_umti": int(mask.sum().item()),
            "num_valid_topo_tokens_per_batch": [
                int(value.item()) for value in mask.sum(dim=1).detach().cpu()
            ],
        }
        if topo_memory_state is not None:
            for key, value in topo_memory_state.get("stats", {}).items():
                if isinstance(value, (int, float)):
                    stats.setdefault(key, value)
        return MemoryPack(
            tokens=tokens,
            mask=mask,
            positions=positions,
            type_ids=type_ids,
            scores=scores,
            stats=stats,
        )

    @staticmethod
    def build_semantic_pack_placeholder_static(batch=None, hidden_size=None, device=None, dtype=None) -> MemoryPack:
        if batch is not None:
            batch_size = int(batch["batch_size"])
            hidden = int(hidden_size or batch["hidden_size"])
            device = batch["device"]
            dtype = batch.get("dtype", torch.float32)
            semantic_type_id = int(batch.get("semantic_node_type_id", SEMANTIC_TYPE_ID))
        else:
            if device is None or hidden_size is None:
                raise ValueError("device and hidden_size are required without batch metadata")
            batch_size = 1
            hidden = int(hidden_size)
            dtype = dtype or torch.float32
            semantic_type_id = SEMANTIC_TYPE_ID
        tokens = torch.zeros(batch_size, 0, hidden, device=device, dtype=dtype)
        mask = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)
        positions = torch.zeros(batch_size, 0, 2, device=device, dtype=dtype)
        type_ids = torch.full((batch_size, 0), semantic_type_id, device=device, dtype=torch.long)
        scores = torch.zeros(batch_size, 0, device=device, dtype=dtype)
        return MemoryPack(
            tokens=tokens,
            mask=mask,
            positions=positions,
            type_ids=type_ids,
            scores=scores,
            stats={
                "semantic_tokens.shape": tuple(tokens.shape),
                "semantic_mask.sum": 0,
                "num_semantic_tokens": 0,
            },
        )

    @staticmethod
    def fuse_memory_packs_static(packs: Iterable[MemoryPack]) -> MemoryPack:
        packs = [pack for pack in packs if pack is not None]
        if not packs:
            raise ValueError("fuse_memory_packs requires at least one pack")
        tokens = torch.cat([pack.tokens for pack in packs], dim=1)
        mask = torch.cat([pack.mask for pack in packs], dim=1)
        positions = torch.cat([pack.positions for pack in packs], dim=1)
        type_ids = torch.cat([pack.type_ids for pack in packs], dim=1)
        scores = None
        if all(pack.scores is not None for pack in packs):
            scores = torch.cat([pack.scores for pack in packs], dim=1)
        stats: Dict[str, object] = {}
        for pack in packs:
            stats.update(pack.stats or {})
        stats.update(
            {
                "memory_tokens.shape": tuple(tokens.shape),
                "memory_mask.sum": int(mask.sum().item()),
            }
        )
        return MemoryPack(tokens=tokens, mask=mask, positions=positions, type_ids=type_ids, scores=scores, stats=stats)

    def build_grid_pack(self, historical_memory, batch=None):
        positions = None if batch is None else batch.get("grid_positions", None)
        pack = self.build_grid_pack_static(historical_memory, positions, self.hidden_size)
        if batch is not None and batch.get("grid_mask", None) is not None:
            grid_mask = batch["grid_mask"].to(device=pack.tokens.device).bool()
            if grid_mask.shape == pack.mask.shape:
                pack.mask = grid_mask
                pack.scores = grid_mask.to(pack.tokens.dtype)
                pack.stats["grid_mask.sum"] = int(grid_mask.sum().item())
                pack.stats["num_grid_tokens"] = int(grid_mask.sum().item())
        pack.tokens = self._project_tokens_if_needed(pack.tokens, "grid_projection")
        return pack

    def build_topo_pack(self, topo_memory_state, batch=None):
        pack = self.build_topo_pack_static(topo_memory_state, batch=batch, hidden_size=self.hidden_size)
        pack.tokens = self._project_tokens_if_needed(pack.tokens, "topo_projection")
        before_stats = self._masked_token_norm_stats(pack.tokens, pack.mask)
        pack.stats["topo_token_norm_before_gate"] = before_stats["mean"]
        pack.stats["topo_token_norm_before_gate_std"] = before_stats["std"]
        gate = torch.sigmoid(self.topo_gate_logit)
        if self.use_topo_gate and pack.tokens.shape[1] > 0:
            valid_mask = pack.mask.unsqueeze(-1).to(pack.tokens.dtype)
            pack.tokens = gate * pack.tokens * valid_mask + pack.tokens * (1.0 - valid_mask)
        pack.stats["topo_gate"] = float(gate.detach().item())
        after_stats = self._masked_token_norm_stats(pack.tokens, pack.mask)
        pack.stats["topo_token_norm_after_gate"] = after_stats["mean"]
        pack.stats["topo_token_norm_after_gate_std"] = after_stats["std"]
        return pack

    def build_semantic_pack_placeholder(self, batch=None):
        if not self._semantic_warning_emitted and _rank0():
            warnings.warn("UMTI semantic nodes are reserved for Stage 3 and currently return an empty pack.")
            self._semantic_warning_emitted = True
        return self.build_semantic_pack_placeholder_static(batch=batch, hidden_size=self.hidden_size)

    def fuse_memory_packs(self, packs):
        return self.fuse_memory_packs_static(packs)

    def forward(self, historical_memory, topo_memory_state=None, batch=None):
        packs = [self.build_grid_pack(historical_memory, batch=batch)]

        if bool(getattr(self.args, "enable_topo_memory", False)):
            packs.append(self.build_topo_pack(topo_memory_state, batch=batch))

        if bool(getattr(self.args, "use_semantic_nodes", False)):
            packs.append(self.build_semantic_pack_placeholder(batch=batch))

        memory_pack = self.fuse_memory_packs(packs)
        tokens = memory_pack.tokens
        tokens = tokens + self.position_mlp(memory_pack.positions.to(tokens.dtype))
        if self.use_memory_type_embedding:
            tokens = tokens + self.memory_type_embedding(memory_pack.type_ids.clamp_min(0))
        tokens = tokens * memory_pack.mask.unsqueeze(-1).to(tokens.dtype)
        memory_pack.tokens = tokens
        grid_norm = self._masked_token_norm_stats(
            tokens,
            (memory_pack.type_ids == GRID_MEMORY_TYPE_ID) & memory_pack.mask,
        )
        topo_norm = self._masked_token_norm_stats(
            tokens,
            (memory_pack.type_ids == TOPO_PLACE_TYPE_ID) & memory_pack.mask,
        )
        memory_norm = self._masked_token_norm_stats(tokens, memory_pack.mask)
        memory_pack.stats.update(
            {
                "use_umti": bool(getattr(self.args, "use_umti", False)),
                "enable_topo_memory": bool(getattr(self.args, "enable_topo_memory", False)),
                "use_time_decay": bool(getattr(self.args, "use_time_decay", False)),
                "use_memory_type_embedding": self.use_memory_type_embedding,
                "use_topo_gate": self.use_topo_gate,
                "use_semantic_nodes": bool(getattr(self.args, "use_semantic_nodes", False)),
                "topo_gate": float(torch.sigmoid(self.topo_gate_logit).detach().item()),
                "num_memory_tokens": int(memory_pack.mask.sum().item()),
                "grid_token_norm_mean": grid_norm["mean"],
                "grid_token_norm_std": grid_norm["std"],
                "topo_token_norm_mean": topo_norm["mean"],
                "topo_token_norm_std": topo_norm["std"],
                "memory_token_norm_mean": memory_norm["mean"],
                "memory_token_norm_std": memory_norm["std"],
                "num_auxiliary_tokens": int(
                    ((memory_pack.type_ids == AUXILIARY_TYPE_ID) & memory_pack.mask).sum().item()
                ),
                # Reserved for ELAM / uncertainty-aware candidate navigation.
                "alignment_confidence": None,
                "alignment_entropy": None,
            }
        )
        self._maybe_print_debug(memory_pack, batch.get("debug_context", {}) if batch is not None else {})
        return memory_pack.to_dict()

    @staticmethod
    def _masked_token_norm_stats(tokens: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        if tokens.numel() == 0 or mask is None or mask.numel() == 0:
            return {"mean": 0.0, "std": 0.0}
        valid_tokens = tokens[mask.bool()]
        if valid_tokens.numel() == 0:
            return {"mean": 0.0, "std": 0.0}
        norms = valid_tokens.norm(dim=-1)
        return {
            "mean": float(norms.mean().detach().item()),
            "std": float(norms.std(unbiased=False).detach().item()) if norms.numel() > 1 else 0.0,
        }

    def _maybe_print_debug(self, memory_pack: MemoryPack, debug_context=None) -> None:
        if not self.debug_memory_tokens or not _rank0():
            return

        debug_context = debug_context or {}
        if debug_context.get("enabled", True) is False:
            return

        interval = max(int(getattr(self.args, "log_every", 1)), 1)
        batch_idx = debug_context.get("batch_idx", None)
        if batch_idx is None:
            self._debug_counter += 1
            should_print = self._debug_counter <= 3 or self._debug_counter % interval == 0
            debug_key_base = ("fallback", self._debug_counter)
        else:
            batch_idx = int(batch_idx)
            should_print = batch_idx <= 3 or batch_idx % interval == 0
        if not should_print:
            return
        current_t = int(debug_context.get("timestep", -1))
        topo_mask_sum = int(memory_pack.stats.get("topo_mask.sum", 0))
        debug_key_base = (
            debug_context.get("epoch", "n/a"),
            debug_context.get("batch_idx", "n/a"),
            debug_context.get("phase", "n/a"),
        )
        if current_t == 0:
            debug_key = debug_key_base + ("initial",)
        elif topo_mask_sum > 0:
            debug_key = debug_key_base + ("first_valid_topo",)
        else:
            return
        if debug_key in self._debug_printed:
            return
        self._debug_printed.add(debug_key)
        stats = memory_pack.stats
        lines = [
            f"[UMTI] epoch={debug_context.get('epoch', 'n/a')} batch={debug_context.get('batch_idx', 'n/a')} phase={debug_context.get('phase', 'n/a')} timestep={debug_context.get('timestep', 'n/a')}",
            f"[UMTI] use_umti={stats.get('use_umti')} enable_topo_memory={stats.get('enable_topo_memory')} use_time_decay={stats.get('use_time_decay')} use_memory_type_embedding={stats.get('use_memory_type_embedding')} use_topo_gate={stats.get('use_topo_gate')}",
            f"[UMTI] grid_tokens.shape={stats.get('grid_tokens.shape')}",
            f"[UMTI] topo_tokens.shape={stats.get('topo_tokens.shape', (memory_pack.tokens.shape[0], 0, self.hidden_size))}",
            f"[UMTI] memory_tokens.shape={tuple(memory_pack.tokens.shape)}",
            f"[UMTI] grid_mask.sum={stats.get('grid_mask.sum', 0)} topo_mask.sum={topo_mask_sum} memory_mask.sum={int(memory_pack.mask.sum().item())}",
            f"[UMTI] global_retrieved_nodes={stats.get('global_retrieved_nodes', 0.0)} local_retrieved_nodes={stats.get('local_retrieved_nodes', 0.0)}",
            f"[UMTI] grid_token_norm mean={stats.get('grid_token_norm_mean', 0.0):.6f} std={stats.get('grid_token_norm_std', 0.0):.6f}",
            f"[UMTI] topo_token_norm mean={stats.get('topo_token_norm_mean', 0.0):.6f} std={stats.get('topo_token_norm_std', 0.0):.6f}",
            f"[UMTI] memory_token_norm mean={stats.get('memory_token_norm_mean', 0.0):.6f} std={stats.get('memory_token_norm_std', 0.0):.6f}",
            f"[UMTI] topo_token_norm_before_gate mean={stats.get('topo_token_norm_before_gate', 0.0):.6f} std={stats.get('topo_token_norm_before_gate_std', 0.0):.6f}",
            f"[UMTI] topo_token_norm_after_gate mean={stats.get('topo_token_norm_after_gate', 0.0):.6f} std={stats.get('topo_token_norm_after_gate_std', 0.0):.6f}",
            f"[UMTI] topo_gate={stats.get('topo_gate'):.6f}",
            f"[UMTI] num_grid_tokens={stats.get('num_grid_tokens', 0)} num_topo_tokens={stats.get('num_topo_tokens', 0)} num_valid_topo_tokens_per_batch={stats.get('num_valid_topo_tokens_per_batch', [])}",
        ]
        if bool(stats.get("enable_topo_memory")) and topo_mask_sum == 0:
            lines.append("[UMTI][Warning] enable_topo_memory=True but no valid topo tokens are passed to UMTI.")
        print("\n".join(lines), flush=True)
