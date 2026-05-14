from dataclasses import dataclass

import torch


GRID_MEMORY_TYPE = 0
TOPO_MEMORY_TYPE = 1
SEMANTIC_MEMORY_TYPE = 2
RESERVED_MEMORY_TYPE = 3


@dataclass
class UnifiedMemoryTokens:
    tokens: torch.Tensor              # [B, N, D]
    padding_mask: torch.Tensor        # [B, N], bool, True means padding
    positions: torch.Tensor           # [B, N, 2], normalized xy
    type_ids: torch.Tensor            # [B, N], long
    cell_to_token_map: torch.Tensor   # [B, G], long
    source: str                       # "grid" / "topo" / future "semantic_topo"
    stats: dict = None
    raw: dict = None

    def validate(self, name="memory"):
        """Validate unified memory tensors and index ranges."""

        if self.tokens.dim() != 3:
            raise RuntimeError(f"{name}.tokens must be 3D [B, N, D], got {tuple(self.tokens.shape)}")
        batch_size, token_count, _ = self.tokens.shape

        if self.padding_mask.shape != (batch_size, token_count):
            raise RuntimeError(
                f"{name}.padding_mask must be [B, N]={batch_size, token_count}, "
                f"got {tuple(self.padding_mask.shape)}"
            )
        if self.padding_mask.dtype != torch.bool:
            raise RuntimeError(f"{name}.padding_mask must have dtype bool, got {self.padding_mask.dtype}")

        if self.positions.shape != (batch_size, token_count, 2):
            raise RuntimeError(
                f"{name}.positions must be [B, N, 2]={batch_size, token_count, 2}, "
                f"got {tuple(self.positions.shape)}"
            )

        if self.type_ids.shape != (batch_size, token_count):
            raise RuntimeError(
                f"{name}.type_ids must be [B, N]={batch_size, token_count}, "
                f"got {tuple(self.type_ids.shape)}"
            )
        if self.type_ids.dtype != torch.long:
            raise RuntimeError(f"{name}.type_ids must have dtype long, got {self.type_ids.dtype}")

        if self.cell_to_token_map.dim() != 2 or self.cell_to_token_map.shape[0] != batch_size:
            raise RuntimeError(
                f"{name}.cell_to_token_map must be [B, G] with B={batch_size}, "
                f"got {tuple(self.cell_to_token_map.shape)}"
            )
        if self.cell_to_token_map.dtype != torch.long:
            raise RuntimeError(
                f"{name}.cell_to_token_map must have dtype long, got {self.cell_to_token_map.dtype}"
            )

        if token_count <= 0:
            raise RuntimeError(f"{name}.tokens must contain at least one token slot")

        if self.cell_to_token_map.numel() > 0:
            map_min = int(self.cell_to_token_map.min().item())
            map_max = int(self.cell_to_token_map.max().item())
            if map_min < 0:
                raise RuntimeError(f"{name}.cell_to_token_map min must be >= 0, got {map_min}")
            if map_max >= token_count:
                raise RuntimeError(
                    f"{name}.cell_to_token_map max must be < N={token_count}, got {map_max}"
                )

        return self

    def valid_mask(self):
        """Return [B, N] mask where True marks real memory tokens."""

        return ~self.padding_mask

    def valid_counts(self):
        """Return [B] valid token counts."""

        return self.valid_mask().sum(dim=1)

    def max_tokens(self):
        """Return padded token capacity N."""

        return int(self.tokens.shape[1])

    def gather_cell_logits(self, token_logits, fill_value=-1e9):
        """Gather [B, N] token logits back to [B, G] grid-cell logits."""

        self.validate("memory")
        if token_logits.dim() != 2:
            raise RuntimeError(
                f"token_logits must be [B, N], got {tuple(token_logits.shape)}"
            )
        batch_size, token_count = token_logits.shape
        if batch_size != self.tokens.shape[0] or token_count != self.tokens.shape[1]:
            raise RuntimeError(
                f"token_logits shape {tuple(token_logits.shape)} does not match "
                f"memory tokens [B, N]={tuple(self.tokens.shape[:2])}"
            )

        if self.cell_to_token_map.numel() > 0:
            map_min = int(self.cell_to_token_map.min().item())
            map_max = int(self.cell_to_token_map.max().item())
            if map_min < 0 or map_max >= token_count:
                raise RuntimeError(
                    "Invalid memory gather index: cell_to_token_map min={}, max={}, token_count={}".format(
                        map_min, map_max, token_count
                    )
                )

        padding_mask = self.padding_mask.to(token_logits.device)
        safe_logits = token_logits.masked_fill(padding_mask, fill_value)
        fully_masked = padding_mask.all(dim=1)
        if fully_masked.any():
            safe_logits = safe_logits.clone()
            safe_logits[fully_masked, 0] = 0.0
        return torch.gather(safe_logits, 1, self.cell_to_token_map.to(token_logits.device))


def memory_stats(memory: UnifiedMemoryTokens) -> dict:
    """Return stats for memory.tokens [B, N, D] and masks [B, N] as Python floats."""

    memory.validate("memory")
    valid = memory.valid_mask()
    total_slots = float(valid.numel())
    valid_count = valid.sum().to(torch.float32)
    padding_ratio = 0.0
    if total_slots > 0:
        padding_ratio = float((memory.padding_mask.sum().to(torch.float32) / total_slots).item())

    def _type_count(type_id):
        return float(((memory.type_ids == type_id) & valid).sum().to(torch.float32).item())

    batch_size = max(int(memory.tokens.shape[0]), 1)
    return {
        "memory_token_count": float(memory.tokens.shape[1]),
        "memory_valid_token_count": float((valid_count / batch_size).item()),
        "memory_padding_ratio": padding_ratio,
        "memory_grid_token_count": _type_count(GRID_MEMORY_TYPE) / batch_size,
        "memory_topo_token_count": _type_count(TOPO_MEMORY_TYPE) / batch_size,
        "memory_semantic_token_count": _type_count(SEMANTIC_MEMORY_TYPE) / batch_size,
    }
