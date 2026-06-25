import math

import torch
from torch import nn
from torch.nn import functional as F


class StopVisualContextAdapter(nn.Module):
    """Lightweight visual context provider for stop visual context diagnostics.

    Inputs:
        visual_tokens: [B, T, 512, N]
        instruction_embedding: [B, 49] legacy lang_cls or [B, embed_dim]

    Outputs:
        stop_visual_context: [B, T, embed_dim]
        diagnostics: detached scalar diagnostics for logging
    """

    def __init__(
        self,
        mode="global_attn",
        visual_dim=512,
        embed_dim=768,
        num_heads=8,
        dropout=0.1,
        instruction_dim=49,
        num_regions=4,
        topk=5,
    ):
        super().__init__()
        if mode not in ("global_attn", "fixed_partition"):
            raise ValueError(f"Unsupported stop visual context mode: {mode}.")
        if mode == "fixed_partition" and int(num_regions) != 4:
            raise NotImplementedError("fixed_partition currently supports only num_regions=4.")

        self.mode = mode
        self.visual_dim = visual_dim
        self.embed_dim = embed_dim
        self.num_regions = int(num_regions)
        self.topk = int(topk)

        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.legacy_instruction_proj = nn.Linear(instruction_dim, embed_dim)
        self.embed_instruction_proj = (
            nn.Identity() if instruction_dim == embed_dim else nn.Linear(embed_dim, embed_dim)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_tokens, instruction_embedding):
        if visual_tokens.dim() != 4:
            raise ValueError(
                "visual_tokens must be [B, T, visual_dim, N], "
                f"got {tuple(visual_tokens.shape)}."
            )
        batch_size, time_steps, visual_dim, num_tokens = visual_tokens.shape
        if visual_dim != self.visual_dim:
            raise ValueError(f"Expected visual_dim={self.visual_dim}, got {visual_dim}.")
        input_grid_size = int(math.sqrt(num_tokens))
        if input_grid_size * input_grid_size != num_tokens:
            input_grid_size = -1

        visual_seq = visual_tokens.permute(0, 1, 3, 2).reshape(
            batch_size * time_steps,
            num_tokens,
            visual_dim,
        )
        visual_seq = self.visual_proj(visual_seq)
        instruction_query = self._project_instruction(
            instruction_embedding,
            repeat_factor=time_steps,
        )

        if self.mode == "global_attn":
            context, attn = self._attend(instruction_query, visual_seq)
            diagnostics = self._attention_diagnostics(attn, prefix="global_attn")
        elif self.mode == "fixed_partition":
            fixed_region_tokens, region_sizes = self._fixed_partition_tokens(visual_seq, num_tokens)
            context, attn = self._attend(instruction_query, fixed_region_tokens)
            diagnostics = self._attention_diagnostics(attn, prefix="fixed_region_select")
            diagnostics.update(self._fixed_region_diagnostics(fixed_region_tokens, region_sizes))
        else:
            raise ValueError(f"Unsupported stop visual context mode: {self.mode}.")

        stop_visual_context = context.reshape(batch_size, time_steps, self.embed_dim)
        with torch.no_grad():
            diagnostics["stop_visual_context_input_tokens"] = float(num_tokens)
            diagnostics["stop_visual_context_input_grid_size"] = float(input_grid_size)
            diagnostics["stop_visual_context_norm"] = float(
                torch.nan_to_num(
                    stop_visual_context.detach().to(torch.float32).norm(dim=-1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).mean().item()
            )
        return stop_visual_context, diagnostics

    def _project_instruction(self, instruction_embedding, repeat_factor):
        if instruction_embedding is None:
            raise ValueError("instruction_embedding is required for StopVisualContextAdapter.")
        if instruction_embedding.dim() != 2:
            raise ValueError(
                "instruction_embedding must be [B, 49] or [B, embed_dim], "
                f"got {tuple(instruction_embedding.shape)}."
            )
        if instruction_embedding.shape[-1] == self.legacy_instruction_proj.in_features:
            instruction = self.legacy_instruction_proj(instruction_embedding)
        elif instruction_embedding.shape[-1] == self.embed_dim:
            instruction = self.embed_instruction_proj(instruction_embedding)
        else:
            raise ValueError(
                "instruction_embedding last dim must be 49 or embed_dim; "
                f"got {instruction_embedding.shape[-1]}."
            )
        if repeat_factor < 1:
            raise ValueError(f"repeat_factor must be >= 1, got {repeat_factor}.")
        return instruction.repeat_interleave(repeat_factor, dim=0)

    def _attend(self, instruction_query, key_value_tokens):
        context, attn = self.attention(
            query=instruction_query.unsqueeze(1),
            key=key_value_tokens,
            value=key_value_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        return self.out_norm(context.squeeze(1)), attn

    def _fixed_partition_tokens(self, visual_seq, num_tokens):
        if self.num_regions != 4:
            raise NotImplementedError("fixed_partition currently supports only num_regions=4.")

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                "fixed_partition requires square spatial tokens, "
                f"got N={num_tokens}."
            )

        device = visual_seq.device
        indices = torch.arange(num_tokens, device=device)
        rows = indices // side
        cols = indices % side
        row_mid = side // 2
        col_mid = side // 2
        masks = (
            (rows < row_mid) & (cols < col_mid),
            (rows < row_mid) & (cols >= col_mid),
            (rows >= row_mid) & (cols < col_mid),
            (rows >= row_mid) & (cols >= col_mid),
        )

        region_tokens = []
        region_sizes = []
        for mask in masks:
            token_indices = torch.nonzero(mask, as_tuple=False).flatten()
            if token_indices.numel() == 0:
                raise ValueError(
                    f"fixed_partition produced an empty region for N={num_tokens}."
                )
            region_sizes.append(int(token_indices.numel()))
            region_tokens.append(visual_seq.index_select(1, token_indices).mean(dim=1))
        return torch.stack(region_tokens, dim=1), region_sizes

    @staticmethod
    def _normalize_attention(attn):
        if attn is None:
            return None
        attn = torch.nan_to_num(
            attn.detach().to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        if attn.dim() != 3 or attn.shape[-2] <= 0 or attn.shape[-1] <= 0:
            return None
        attn = attn.squeeze(-2) if attn.shape[-2] == 1 else attn.mean(dim=-2)
        attn = attn.clamp(min=1e-8)
        return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _attention_diagnostics(self, attn, prefix):
        eps = 1e-6
        with torch.no_grad():
            p = self._normalize_attention(attn)
            if p is None:
                return {}

            num_items = p.shape[-1]
            entropy_raw = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
            if num_items > 1:
                entropy = entropy_raw / math.log(num_items)
            else:
                entropy = entropy_raw.new_zeros(entropy_raw.shape)

            diagnostics = {
                f"{prefix}_entropy": float(
                    torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
                ),
                f"{prefix}_max": float(
                    torch.nan_to_num(p.max(dim=-1).values, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
                ),
                f"{prefix}_effective_num": float(
                    torch.nan_to_num(
                        torch.exp(entropy_raw),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).mean().item()
                ),
            }

            topk = min(max(int(self.topk), 1), num_items)
            topk_mass = torch.topk(p, k=topk, dim=-1).values.sum(dim=-1)
            diagnostics[f"{prefix}_topk_mass"] = float(
                torch.nan_to_num(topk_mass, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
            )
            if num_items > 1:
                top2 = torch.topk(p, k=2, dim=-1).values
                peak_margin = top2[:, 0] - top2[:, 1]
            else:
                peak_margin = p.new_zeros(p.shape[:-1])
            diagnostics[f"{prefix}_peak_margin"] = float(
                torch.nan_to_num(peak_margin, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
            )
            return diagnostics

    @staticmethod
    def _fixed_region_diagnostics(fixed_region_tokens, region_sizes):
        eps = 1e-6
        with torch.no_grad():
            tokens = torch.nan_to_num(
                fixed_region_tokens.detach().to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            token_norm = F.normalize(tokens, dim=-1, eps=eps)
            sim = torch.matmul(token_norm, token_norm.transpose(-1, -2))
            num_regions = sim.shape[-1]
            offdiag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=sim.device)
            offdiag_sim = sim[..., offdiag_mask]
            size_tensor = torch.tensor(region_sizes, dtype=torch.float32)
            return {
                "fixed_region_token_offdiag_cos": float(
                    torch.nan_to_num(offdiag_sim, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
                ),
                "fixed_partition_region_sizes": list(region_sizes),
                "fixed_partition_region_size_min": float(size_tensor.min().item()),
                "fixed_partition_region_size_max": float(size_tensor.max().item()),
                "fixed_partition_region_size_mean": float(size_tensor.mean().item()),
            }
