import math

import torch
from torch import nn
from torch.nn import functional as F


class RegionPromptAdapter(nn.Module):
    """Learned region-query adapter for DarkNet visual tokens.

    Inputs:
        visual_tokens:
            [B, T, 512, N] for a sequence of frames, or [B, 512, N] for one frame.
            The N dimension is the raw visual-token grid; current DarkNet features use N=49 (7x7).
        instruction_embedding:
            Optional [B, 768] instruction embedding. For compatibility with the current
            HETT baseline, [B, 49] legacy lang_cls vectors are also accepted and projected.

    Outputs:
        region_tokens:
            [B, T, K, 768] for sequence input, or [B, K, 768] for single-frame input.
            K is the number of learned region queries.
    """

    def __init__(
        self,
        visual_dim=512,
        embed_dim=768,
        num_region_queries=4,
        num_heads=8,
        dropout=0.0,
        instruction_dim=768,
        legacy_lang_dim=49,
        condition_generation=False,
        fuse_instruction=False,
        query_init="random",
        query_scale=0.1,
        use_pos_embed=False,
        max_spatial_tokens=25,
        attn_topk=5,
    ):
        super().__init__()
        if not 4 <= num_region_queries <= 8:
            raise ValueError("num_region_queries must be in the range [4, 8].")
        if query_init not in ("random", "orthogonal", "pos"):
            raise ValueError(f"query_init must be 'random', 'orthogonal', or 'pos', got {query_init}.")
        if max_spatial_tokens < 1:
            raise ValueError(f"max_spatial_tokens must be positive, got {max_spatial_tokens}.")
        if query_init == "pos" and num_region_queries > max_spatial_tokens:
            raise ValueError(
                "query_init='pos' requires num_region_queries <= max_spatial_tokens, "
                f"got {num_region_queries} > {max_spatial_tokens}."
            )

        self.visual_dim = visual_dim
        self.embed_dim = embed_dim
        self.num_region_queries = num_region_queries
        self.instruction_dim = instruction_dim
        self.condition_generation = condition_generation
        self.fuse_instruction = fuse_instruction
        self.query_init = query_init
        self.query_scale = query_scale
        self.use_pos_embed = use_pos_embed
        self.max_spatial_tokens = max_spatial_tokens
        self.attn_topk = attn_topk
        self.latest_visual_token_offdiag_cos = None
        self.latest_raw_visual_token_offdiag_cos = None
        self.latest_generation_attention_shape = None
        self.latest_region_attn_diversity_loss = None
        self.latest_generation_attention_diagnostics = {}
        self.latest_projected_query_diagnostics = {}
        self.latest_query_spatial_affinity_diagnostics = {}
        self.latest_input_diagnostics = {}

        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, max_spatial_tokens, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        if query_init == "orthogonal":
            self.register_buffer(
                "region_query_pos_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            region_queries = torch.empty(num_region_queries, embed_dim)
            nn.init.orthogonal_(region_queries)
        elif query_init == "pos":
            indices = self._get_pos_init_indices(num_region_queries, max_spatial_tokens)
            self.register_buffer("region_query_pos_indices", indices.clone(), persistent=False)
            region_queries = self.spatial_pos_embed.detach()[0, indices, :].clone()
            region_queries = F.normalize(region_queries, dim=-1, eps=1e-6) * (
                query_scale * math.sqrt(embed_dim)
            )
        else:
            self.register_buffer(
                "region_query_pos_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            region_queries = torch.randn(num_region_queries, embed_dim)
        if query_init != "pos":
            region_queries = region_queries * query_scale
        self.region_queries = nn.Parameter(region_queries)
        self.instruction_proj = (
            nn.Linear(instruction_dim, embed_dim)
            if instruction_dim is not None and instruction_dim != embed_dim
            else nn.Identity()
        )
        self.embed_instruction_proj = nn.Linear(embed_dim, embed_dim)
        self.legacy_instruction_proj = (
            nn.Linear(legacy_lang_dim, embed_dim)
            if legacy_lang_dim != instruction_dim
            else self.instruction_proj
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.region_selection = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)

        # Compatibility path for the existing ET_haa.fc2: region tokens can be summarized
        # back to a [B, 49] frame feature before the original fc2 maps it to 768 dims.
        self.legacy_grid_proj = nn.Linear(embed_dim, legacy_lang_dim)

    def _project_instruction(self, instruction_embedding, repeat_factor=1):
        if instruction_embedding is None:
            return None

        if instruction_embedding.dim() != 2:
            raise ValueError(
                "instruction_embedding must be [B, instruction_dim], [B, 768], or legacy [B, 49], "
                f"got {tuple(instruction_embedding.shape)}."
            )

        if self.instruction_dim is not None and instruction_embedding.shape[-1] == self.instruction_dim:
            instruction = self.instruction_proj(instruction_embedding)
        elif instruction_embedding.shape[-1] == self.embed_dim:
            instruction = self.embed_instruction_proj(instruction_embedding)
        elif (
            isinstance(self.legacy_instruction_proj, nn.Linear)
            and instruction_embedding.shape[-1] == self.legacy_instruction_proj.in_features
        ):
            instruction = self.legacy_instruction_proj(instruction_embedding)
        else:
            raise ValueError(
                "instruction_embedding last dim must match instruction_dim, 768, or 49; "
                f"got {instruction_embedding.shape[-1]}."
            )

        if repeat_factor < 1:
            raise ValueError(f"repeat_factor must be >= 1, got {repeat_factor}.")
        return instruction.repeat_interleave(repeat_factor, dim=0)

    @staticmethod
    def _mean_offdiag_cos(tokens):
        if tokens.dim() != 3 or tokens.shape[1] <= 1:
            return None

        eps = 1e-6
        with torch.no_grad():
            tokens = torch.nan_to_num(
                tokens.detach().to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            token_norm = F.normalize(tokens, dim=-1, eps=eps)
            sim = torch.matmul(token_norm, token_norm.transpose(-1, -2))
            num_tokens = sim.shape[-1]
            offdiag_mask = ~torch.eye(num_tokens, dtype=torch.bool, device=sim.device)
            offdiag_sim = sim[..., offdiag_mask]
            if offdiag_sim.numel() == 0:
                return None
            value = torch.nan_to_num(offdiag_sim, nan=0.0, posinf=0.0, neginf=0.0).mean()
            return float(value.item())

    @staticmethod
    def _get_pos_init_indices(num_region_queries, max_spatial_tokens):
        if num_region_queries > max_spatial_tokens:
            raise ValueError(
                "num_region_queries must be <= max_spatial_tokens for position initialization, "
                f"got {num_region_queries} > {max_spatial_tokens}."
            )

        grid_side = int(math.sqrt(max_spatial_tokens))
        if grid_side * grid_side == max_spatial_tokens and num_region_queries == 4:
            top_left = 0
            top_right = grid_side - 1
            bottom_left = grid_side * (grid_side - 1)
            bottom_right = grid_side * grid_side - 1
            return torch.tensor(
                [
                    top_left,
                    top_right,
                    bottom_left,
                    bottom_right,
                ],
                dtype=torch.long,
            )

        return torch.linspace(0, max_spatial_tokens - 1, steps=num_region_queries).long()

    @staticmethod
    def _input_grid_size(num_visual_tokens):
        grid_side = int(math.sqrt(num_visual_tokens))
        if grid_side * grid_side == num_visual_tokens:
            return grid_side
        return -1

    def _set_input_diagnostics(self, num_visual_tokens):
        self.latest_input_diagnostics = {
            "region_prompt_input_tokens": float(num_visual_tokens),
            "region_prompt_input_grid_size": float(self._input_grid_size(num_visual_tokens)),
            "region_prompt_max_spatial_tokens": float(self.max_spatial_tokens),
        }

    def _validate_spatial_token_count(self, num_visual_tokens):
        if num_visual_tokens > self.max_spatial_tokens:
            raise ValueError(
                "RegionPrompt spatial position embedding is too short for visual tokens: "
                f"N={num_visual_tokens}, max_spatial_tokens={self.max_spatial_tokens}. "
                "Set --region_prompt_max_spatial_tokens to at least the raw visual token count."
            )

    def _add_spatial_pos_embed(self, visual_seq):
        num_visual_tokens = visual_seq.size(1)
        self._validate_spatial_token_count(num_visual_tokens)
        return visual_seq + self.spatial_pos_embed[:, :num_visual_tokens, :].to(
            device=visual_seq.device,
            dtype=visual_seq.dtype,
        )

    def spatial_pos_diagnostics(self):
        with torch.no_grad():
            pos = torch.nan_to_num(
                self.spatial_pos_embed.detach().to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            return {
                "spatial_pos_embed_norm": float(pos.norm(dim=-1).mean().item()),
                "spatial_pos_embed_offdiag_cos": self._mean_offdiag_cos(pos),
            }

    @staticmethod
    def _normalize_generation_attention(attn):
        if attn is None:
            return None

        attn = torch.nan_to_num(
            attn.to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        if attn.dim() != 3 or attn.shape[-2] <= 0 or attn.shape[-1] <= 0:
            return None

        attn = attn.clamp(min=1e-8)
        return attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def generation_attention_diversity_loss(attn, mode="cosine_square"):
        if mode != "cosine_square":
            raise ValueError(f"Unsupported region attention diversity mode: {mode}")
        if attn is None:
            return None

        normalized_attn = RegionPromptAdapter._normalize_generation_attention(attn)
        if normalized_attn is None:
            return attn.sum() * 0.0
        if normalized_attn.shape[-2] <= 1:
            return normalized_attn.sum() * 0.0

        attn_norm = F.normalize(normalized_attn, dim=-1, eps=1e-6)
        sim = torch.matmul(attn_norm, attn_norm.transpose(-1, -2))
        num_regions = sim.shape[-1]
        offdiag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=sim.device)
        offdiag_sim = sim[:, offdiag_mask]
        if offdiag_sim.numel() == 0:
            return normalized_attn.sum() * 0.0
        return (offdiag_sim ** 2).mean()

    @staticmethod
    def _attention_map_diagnostics(attn, topk=5):
        if attn is None:
            return {}

        eps = 1e-6
        with torch.no_grad():
            p = RegionPromptAdapter._normalize_generation_attention(attn.detach())
            if p is None:
                return {}

            num_visual_tokens = p.shape[-1]

            entropy_raw = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
            entropy = entropy_raw
            if num_visual_tokens > 1:
                entropy = entropy / math.log(num_visual_tokens)
            else:
                entropy = entropy.new_zeros(entropy.shape)

            diagnostics = {
                "region_gen_attn_entropy": float(
                    torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
                ),
                "region_gen_attn_max": float(
                    torch.nan_to_num(p.max(dim=-1).values, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
                ),
                "region_gen_attn_effective_num": float(
                    torch.nan_to_num(
                        torch.exp(entropy_raw),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).mean().item()
                ),
            }
            topk = min(max(int(topk), 1), num_visual_tokens)
            topk_mass = torch.topk(p, k=topk, dim=-1).values.sum(dim=-1)
            diagnostics["region_gen_attn_topk_mass"] = float(
                torch.nan_to_num(topk_mass, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
            )
            if num_visual_tokens > 1:
                top2 = torch.topk(p, k=2, dim=-1).values
                peak_margin = top2[..., 0] - top2[..., 1]
            else:
                peak_margin = p.new_zeros(p.shape[:-1])
            diagnostics["region_gen_attn_peak_margin"] = float(
                torch.nan_to_num(peak_margin, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
            )

            if p.shape[-2] > 1:
                attn_norm = F.normalize(p, dim=-1, eps=eps)
                sim = torch.matmul(attn_norm, attn_norm.transpose(-1, -2))
                num_regions = sim.shape[-1]
                offdiag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=sim.device)
                offdiag_sim = sim[..., offdiag_mask]
                diagnostics["region_gen_attn_offdiag_cos"] = float(
                    torch.nan_to_num(offdiag_sim, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
                )

            return diagnostics

    def _projected_query_diagnostics(self, queries):
        eps = 1e-6
        with torch.no_grad():
            projected_q = None
            embed_dim = self.cross_attention.embed_dim
            in_proj_weight = getattr(self.cross_attention, "in_proj_weight", None)
            if in_proj_weight is not None:
                in_proj_bias = getattr(self.cross_attention, "in_proj_bias", None)
                q_weight = in_proj_weight[:embed_dim, :]
                q_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
                projected_q = F.linear(queries.detach(), q_weight.detach(), q_bias.detach() if q_bias is not None else None)
            else:
                q_proj_weight = getattr(self.cross_attention, "q_proj_weight", None)
                if q_proj_weight is not None:
                    in_proj_bias = getattr(self.cross_attention, "in_proj_bias", None)
                    q_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
                    projected_q = F.linear(
                        queries.detach(),
                        q_proj_weight.detach(),
                        q_bias.detach() if q_bias is not None else None,
                    )

            if projected_q is None or projected_q.dim() != 3:
                return {}

            projected_q = torch.nan_to_num(
                projected_q.to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            diagnostics = {
                "projected_query_norm": float(projected_q.norm(dim=-1).mean().item()),
            }
            if projected_q.shape[-2] > 1:
                projected_q_norm = F.normalize(projected_q, dim=-1, eps=eps)
                sim = torch.matmul(projected_q_norm, projected_q_norm.transpose(-1, -2))
                num_queries = sim.shape[-1]
                offdiag_mask = ~torch.eye(num_queries, dtype=torch.bool, device=sim.device)
                offdiag_sim = sim[..., offdiag_mask]
                diagnostics["projected_query_offdiag_cos"] = float(
                    torch.nan_to_num(offdiag_sim, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
                )

            return diagnostics

    def _query_spatial_affinity_diagnostics(self, queries, visual_seq):
        eps = 1e-6
        with torch.no_grad():
            if queries.dim() != 3 or visual_seq.dim() != 3:
                return {}
            if queries.shape[0] != visual_seq.shape[0] or queries.shape[-1] != visual_seq.shape[-1]:
                return {}

            queries = torch.nan_to_num(
                queries.detach().to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            visual_seq = torch.nan_to_num(
                visual_seq.detach().to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            queries_norm = F.normalize(queries, dim=-1, eps=eps)
            visual_norm = F.normalize(visual_seq, dim=-1, eps=eps)
            affinity = torch.matmul(queries_norm, visual_norm.transpose(-1, -2))
            affinity = torch.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)

            diagnostics = {
                "query_spatial_affinity_mean": float(affinity.mean().item()),
                "query_spatial_affinity_max": float(affinity.max(dim=-1).values.mean().item()),
            }

            if self.query_init == "pos":
                pos_indices = self._get_pos_init_indices(
                    queries.shape[1],
                    visual_seq.shape[1],
                ).to(device=affinity.device, dtype=torch.long)
                query_indices = torch.arange(queries.shape[1], device=affinity.device)
                target_affinity = affinity[:, query_indices, pos_indices]
                target = torch.nan_to_num(target_affinity, nan=0.0, posinf=0.0, neginf=0.0).mean()
                diagnostics["query_spatial_affinity_target"] = float(target.item())
                diagnostics["query_spatial_affinity_gap"] = float((target - affinity.mean()).item())

            return diagnostics

    def region_query_diagnostics(self):
        with torch.no_grad():
            queries = torch.nan_to_num(
                self.region_queries.detach().to(torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            query_norm = float(queries.norm(dim=-1).mean().item())
            query_offdiag_cos = self._mean_offdiag_cos(queries.unsqueeze(0))
        diagnostics = {
            "region_query_norm": query_norm,
            "region_query_offdiag_cos": query_offdiag_cos,
        }
        diagnostics.update(self.spatial_pos_diagnostics())
        return diagnostics

    def forward(
        self,
        visual_tokens,
        instruction_embedding=None,
        compute_attention_diversity=False,
        attention_diversity_mode="cosine_square",
    ):
        self.latest_visual_token_offdiag_cos = None
        self.latest_raw_visual_token_offdiag_cos = None
        self.latest_generation_attention_shape = None
        self.latest_region_attn_diversity_loss = None
        self.latest_generation_attention_diagnostics = {}
        self.latest_projected_query_diagnostics = {}
        self.latest_query_spatial_affinity_diagnostics = {}
        self.latest_input_diagnostics = {}

        single_frame = visual_tokens.dim() == 3
        if single_frame:
            visual_tokens = visual_tokens.unsqueeze(1)
        if visual_tokens.dim() != 4:
            raise ValueError(
                "visual_tokens must be [B, T, 512, N] or [B, 512, N], "
                f"got {tuple(visual_tokens.shape)}."
            )

        batch_size, time_steps, channels, grid_size = visual_tokens.shape
        if channels != self.visual_dim:
            raise ValueError(f"Expected visual channel dim {self.visual_dim}, got {channels}.")
        self._set_input_diagnostics(grid_size)
        self._validate_spatial_token_count(grid_size)

        # [B, T, 512, N] -> [B*T, N, 512] -> [B*T, N, 768]
        visual_seq = visual_tokens.permute(0, 1, 3, 2).reshape(
            batch_size * time_steps,
            grid_size,
            channels,
        )
        self.latest_raw_visual_token_offdiag_cos = self._mean_offdiag_cos(visual_seq)
        visual_seq = self.visual_proj(visual_seq)
        if self.use_pos_embed:
            visual_seq = self._add_spatial_pos_embed(visual_seq)
        self.latest_visual_token_offdiag_cos = self._mean_offdiag_cos(visual_seq)

        # Learned region prompts query the projected visual grid tokens.
        queries = self.region_queries.unsqueeze(0).expand(batch_size * time_steps, -1, -1)
        instruction = None
        if self.condition_generation or self.fuse_instruction:
            instruction = self._project_instruction(instruction_embedding, repeat_factor=time_steps)
        if self.condition_generation and instruction is not None:
            queries = queries + instruction.unsqueeze(1)

        self.latest_projected_query_diagnostics = self._projected_query_diagnostics(queries)
        self.latest_query_spatial_affinity_diagnostics = self._query_spatial_affinity_diagnostics(
            queries,
            visual_seq,
        )
        try:
            region_tokens, gen_attn = self.cross_attention(
                query=queries,
                key=visual_seq,
                value=visual_seq,
                need_weights=True,
                average_attn_weights=False,
            )
        except TypeError:
            region_tokens, gen_attn = self.cross_attention(
                query=queries,
                key=visual_seq,
                value=visual_seq,
                need_weights=True,
            )
        if gen_attn is not None:
            self.latest_generation_attention_shape = tuple(gen_attn.shape)
            if compute_attention_diversity:
                self.latest_region_attn_diversity_loss = self.generation_attention_diversity_loss(
                    gen_attn,
                    mode=attention_diversity_mode,
                )
        self.latest_generation_attention_diagnostics = self._attention_map_diagnostics(
            gen_attn,
            topk=self.attn_topk,
        )

        if self.fuse_instruction and instruction is not None:
            gate = self.fusion_gate(
                torch.cat((region_tokens, instruction.unsqueeze(1).expand_as(region_tokens)), dim=-1)
            )
            region_tokens = region_tokens + gate * instruction.unsqueeze(1)

        region_tokens = self.out_norm(region_tokens)
        region_tokens = region_tokens.view(
            batch_size,
            time_steps,
            self.num_region_queries,
            self.embed_dim,
        )

        if single_frame:
            return region_tokens.squeeze(1)
        return region_tokens

    def to_legacy_frame_feature(self, region_tokens):
        """Pool region tokens to the legacy [B, 49] or [B, T, 49] feature used by fc2."""
        return self.legacy_grid_proj(region_tokens.mean(dim=-2))

    def select_region_context(self, region_tokens, instruction_embedding):
        """Select instruction-relevant region context.

        Args:
            region_tokens: [B, K, 768] or [B, T, K, 768].
            instruction_embedding: [B, 768] or the configured instruction dim
                used by the current HETT baseline, e.g. lang_cls [B, 49].

        Returns:
            region_context: [B, 768] for non-temporal input, or [B, T, 768].
            region_attn: [B, 1, K] for non-temporal input, or [B, T, 1, K].
        """
        is_temporal = region_tokens.dim() == 4
        if is_temporal:
            batch_size, time_steps, num_regions, embed_dim = region_tokens.shape
            flat_region_tokens = region_tokens.reshape(batch_size * time_steps, num_regions, embed_dim)
            repeat_factor = time_steps
        elif region_tokens.dim() == 3:
            batch_size, num_regions, embed_dim = region_tokens.shape
            time_steps = None
            flat_region_tokens = region_tokens
            repeat_factor = 1
        else:
            raise ValueError(
                "region_tokens must be [B, K, 768] or [B, T, K, 768], "
                f"got {tuple(region_tokens.shape)}."
            )

        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected region token dim {self.embed_dim}, got {embed_dim}.")

        instruction = self._project_instruction(instruction_embedding, repeat_factor=repeat_factor)
        if instruction is None:
            raise ValueError("instruction_embedding is required for region selection.")

        query = instruction.unsqueeze(1)
        selected, region_attn = self.region_selection(
            query=query,
            key=flat_region_tokens,
            value=flat_region_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        selected = self.context_norm(selected.squeeze(1))

        if is_temporal:
            selected = selected.view(batch_size, time_steps, self.embed_dim)
            region_attn = region_attn.view(batch_size, time_steps, 1, num_regions)

        return selected, region_attn
