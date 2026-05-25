"""Entity-Landmark Alignment Module v1.

ELAM-v1 is intentionally lightweight: language-role queries align to the
existing UMTI memory-token pack and expose auxiliary alignment losses without
changing the default HETT/UMTI policy path.
"""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _rank0():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


class ELAMv1(nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.args = args
        self.hidden_size = int(hidden_size)
        self.num_roles = int(getattr(args, "num_elam_roles", 4))
        self.dropout = float(getattr(args, "elam_dropout", 0.1))
        num_heads = int(getattr(args, "elam_num_heads", 8))
        if self.hidden_size % num_heads != 0:
            raise ValueError(
                f"ELAM hidden_size={self.hidden_size} must be divisible by elam_num_heads={num_heads}."
            )

        self.role_queries = nn.Parameter(torch.randn(self.num_roles, self.hidden_size) * 0.02)
        self.text_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.memory_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.role_norm = nn.LayerNorm(self.hidden_size)
        self.context_norm = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self._warned_metric_skip = False
        self._warned_spatial_skip = False

    @staticmethod
    def _valid_mask_to_padding_mask(mask, name):
        if mask is None:
            return None
        mask = mask.bool()
        if mask.dim() != 2:
            raise ValueError(f"ELAM expected {name} to have shape [B, N], got {tuple(mask.shape)}")

        # Tokenizer-style masks use True/1 for valid tokens. Padding masks use
        # True for padding. Since BERT-style instructions always have a valid
        # first token, invert only when the first column strongly suggests the
        # input is already a padding mask.
        first_col_valid_rate = mask[:, 0].to(torch.float32).mean().item() if mask.shape[1] > 0 else 1.0
        valid_mask = ~mask if first_col_valid_rate < 0.5 else mask
        return ~valid_mask

    def _zeros(self, like):
        return like.new_zeros(())

    def _warn_once(self, attr, message):
        if not getattr(self, attr) and _rank0():
            print(message)
            setattr(self, attr, True)

    def _grid_logits_and_positions(self, target_prior_logits, memory_positions, memory_mask, memory_type_ids):
        if memory_type_ids is None:
            grid_mask = memory_mask.bool()
        else:
            grid_mask = memory_mask.bool() & (memory_type_ids.long() == 0)

        grid_counts = grid_mask.sum(dim=1)
        if grid_counts.numel() == 0 or int(grid_counts.min().item()) == 0:
            return None, None, grid_mask
        if not torch.equal(grid_counts, grid_counts[:1].expand_as(grid_counts)):
            return None, None, grid_mask

        count = int(grid_counts[0].item())
        grid_logits = torch.stack([target_prior_logits[b][grid_mask[b]] for b in range(target_prior_logits.shape[0])], dim=0)
        grid_positions = None
        if memory_positions is not None:
            grid_positions = torch.stack([memory_positions[b][grid_mask[b]] for b in range(memory_positions.shape[0])], dim=0)
            if grid_positions.shape[1] != count:
                grid_positions = None
        return grid_logits, grid_positions, grid_mask

    def _metric_cell_loss(self, grid_logits, target_cell_labels, grid_shape):
        if target_cell_labels is None:
            self._warn_once(
                "_warned_metric_skip",
                "[ELAM][Warning] metric_cell loss skipped because target_cell_labels are unavailable.",
            )
            return self._zeros(grid_logits)

        target_cell = target_cell_labels.to(device=grid_logits.device).long().view(-1)
        if target_cell.shape[0] != grid_logits.shape[0]:
            self._warn_once(
                "_warned_metric_skip",
                "[ELAM][Warning] metric_cell loss skipped because target_cell_labels batch size is invalid.",
            )
            return self._zeros(grid_logits)

        expected_cells = None
        if grid_shape is not None:
            if isinstance(grid_shape, int):
                expected_cells = int(grid_shape) * int(grid_shape)
            elif len(grid_shape) >= 2:
                expected_cells = int(grid_shape[0]) * int(grid_shape[1])
        if expected_cells is not None and grid_logits.shape[1] != expected_cells:
            self._warn_once(
                "_warned_metric_skip",
                "[ELAM][Warning] metric_cell loss skipped because grid tokens do not match grid_shape.",
            )
            return self._zeros(grid_logits)
        if int(target_cell.min().item()) < 0 or int(target_cell.max().item()) >= grid_logits.shape[1]:
            self._warn_once(
                "_warned_metric_skip",
                "[ELAM][Warning] metric_cell loss skipped because target_cell_labels are out of range.",
            )
            return self._zeros(grid_logits)
        return F.cross_entropy(grid_logits, target_cell)

    def _soft_spatial_loss(self, grid_logits, grid_positions, target_positions):
        if target_positions is None:
            self._warn_once(
                "_warned_spatial_skip",
                "[ELAM][Warning] soft_spatial loss skipped because target_positions are unavailable.",
            )
            return self._zeros(grid_logits)
        if grid_positions is None:
            self._warn_once(
                "_warned_spatial_skip",
                "[ELAM][Warning] soft_spatial loss skipped because grid memory_positions are unavailable.",
            )
            return self._zeros(grid_logits)

        target_positions = target_positions.to(device=grid_logits.device, dtype=grid_positions.dtype)
        if target_positions.dim() != 2 or target_positions.shape[-1] != 2:
            self._warn_once(
                "_warned_spatial_skip",
                "[ELAM][Warning] soft_spatial loss skipped because target_positions shape is invalid.",
            )
            return self._zeros(grid_logits)

        sigma = max(float(getattr(self.args, "soft_spatial_sigma", 0.15)), 1e-6)
        dist = torch.norm(grid_positions - target_positions.unsqueeze(1), dim=-1)
        soft_label = torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))
        soft_label = soft_label / soft_label.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.kl_div(F.log_softmax(grid_logits, dim=-1), soft_label, reduction="batchmean")

    def _query_div_loss(self, role_context):
        if role_context.shape[1] <= 1:
            return self._zeros(role_context)
        normalized = F.normalize(role_context, p=2, dim=-1)
        sim = torch.matmul(normalized, normalized.transpose(1, 2))
        off_diag = ~torch.eye(role_context.shape[1], device=role_context.device, dtype=torch.bool)
        return sim[:, off_diag].pow(2).mean()

    def forward(
        self,
        instruction_tokens,
        instruction_mask,
        memory_tokens,
        memory_mask,
        memory_positions=None,
        memory_type_ids=None,
        target_cell_labels=None,
        target_positions=None,
        grid_shape=None,
    ):
        if memory_tokens is None:
            raise ValueError("ELAM requires memory_tokens from UMTI.")
        if memory_mask is None:
            raise ValueError("ELAM requires memory_mask from UMTI.")
        if memory_mask.bool().sum().item() == 0:
            raise ValueError("ELAM received memory_mask.sum()==0; no valid UMTI memory tokens are available.")

        batch_size = instruction_tokens.shape[0]
        memory_mask = memory_mask.bool()
        instruction_padding_mask = self._valid_mask_to_padding_mask(instruction_mask, "instruction_mask")
        memory_padding_mask = ~memory_mask

        role_queries = self.role_queries.unsqueeze(0).expand(batch_size, -1, -1)
        text_roles, _ = self.text_attention(
            role_queries,
            instruction_tokens,
            instruction_tokens,
            key_padding_mask=instruction_padding_mask,
            need_weights=False,
        )
        text_aware_roles = self.role_norm(role_queries + self.dropout_layer(text_roles))

        memory_context, _ = self.memory_attention(
            text_aware_roles,
            memory_tokens,
            memory_tokens,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        role_context = self.context_norm(text_aware_roles + self.dropout_layer(memory_context))

        alignment_logits = torch.matmul(role_context, memory_tokens.transpose(1, 2)) / math.sqrt(self.hidden_size)
        alignment_logits = alignment_logits.masked_fill(~memory_mask.unsqueeze(1), -1e4)
        alignment_probs = torch.softmax(alignment_logits, dim=-1)
        alignment_probs = alignment_probs.masked_fill(~memory_mask.unsqueeze(1), 0.0)
        alignment_probs = alignment_probs / alignment_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned_memory_tokens = torch.matmul(alignment_probs, memory_tokens)

        target_prior_logits = alignment_logits[:, 0, :]
        target_probs = alignment_probs[:, 0, :]
        alignment_confidence = target_probs.max(dim=-1).values
        alignment_entropy = -(target_probs * torch.log(target_probs.clamp_min(1e-8))).sum(dim=-1)

        grid_logits, grid_positions, grid_mask = self._grid_logits_and_positions(
            target_prior_logits,
            memory_positions,
            memory_mask,
            memory_type_ids,
        )
        if grid_logits is None:
            zero = self._zeros(memory_tokens)
            self._warn_once(
                "_warned_metric_skip",
                "[ELAM][Warning] metric_cell loss skipped because valid grid tokens are unavailable or ragged.",
            )
            self._warn_once(
                "_warned_spatial_skip",
                "[ELAM][Warning] soft_spatial loss skipped because valid grid tokens are unavailable or ragged.",
            )
            metric_cell_loss = zero
            soft_spatial_loss = zero
        else:
            metric_cell_loss = self._metric_cell_loss(grid_logits, target_cell_labels, grid_shape)
            soft_spatial_loss = self._soft_spatial_loss(grid_logits, grid_positions, target_positions)
        query_div_loss = self._query_div_loss(role_context)

        aux_losses = {
            "metric_cell": metric_cell_loss,
            "soft_spatial": soft_spatial_loss,
            "query_div": query_div_loss,
        }
        stats = {
            "elam_step_alignment_confidence": float(alignment_confidence.detach().mean().item()),
            "elam_step_alignment_entropy": float(alignment_entropy.detach().mean().item()),
            "elam_step_metric_cell_loss": float(metric_cell_loss.detach().item()),
            "elam_step_soft_spatial_loss": float(soft_spatial_loss.detach().item()),
            "elam_step_query_div_loss": float(query_div_loss.detach().item()),
            "num_elam_roles": float(self.num_roles),
            "elam_grid_tokens": float(grid_mask.sum(dim=1).to(torch.float32).mean().item()),
        }

        if bool(getattr(self.args, "debug_elam", False)) and _rank0():
            unique_types = (
                torch.unique(memory_type_ids.detach()).cpu().tolist()
                if memory_type_ids is not None
                else ["none"]
            )
            conf_std = alignment_confidence.detach().std(unbiased=False).item()
            ent_std = alignment_entropy.detach().std(unbiased=False).item()
            print(
                "[ELAM]\n"
                f"instruction_tokens.shape={tuple(instruction_tokens.shape)}\n"
                f"memory_tokens.shape={tuple(memory_tokens.shape)}\n"
                f"memory_mask.sum={int(memory_mask.sum().item())}\n"
                f"memory_type_ids unique={unique_types}\n"
                f"alignment_logits.shape={tuple(alignment_logits.shape)}\n"
                f"alignment_probs.shape={tuple(alignment_probs.shape)}\n"
                f"target_prior_logits.shape={tuple(target_prior_logits.shape)}\n"
                f"alignment_confidence mean={alignment_confidence.detach().mean().item():.6f} std={conf_std:.6f}\n"
                f"alignment_entropy mean={alignment_entropy.detach().mean().item():.6f} std={ent_std:.6f}\n"
                f"metric_cell_loss={metric_cell_loss.detach().item():.6f}\n"
                f"soft_spatial_loss={soft_spatial_loss.detach().item():.6f}\n"
                f"query_div_loss={query_div_loss.detach().item():.6f}"
            )

        return {
            "aligned_memory_tokens": aligned_memory_tokens,
            "role_context": role_context,
            "alignment_logits": alignment_logits,
            "alignment_probs": alignment_probs,
            "target_prior_logits": target_prior_logits,
            "alignment_confidence": alignment_confidence,
            "alignment_entropy": alignment_entropy,
            "aux_losses": aux_losses,
            "stats": stats,
        }
