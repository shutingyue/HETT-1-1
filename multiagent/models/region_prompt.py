import torch
from torch import nn


class RegionPromptAdapter(nn.Module):
    """Learned region-query adapter for DarkNet visual tokens.

    Inputs:
        visual_tokens:
            [B, T, 512, 49] for a sequence of frames, or [B, 512, 49] for one frame.
            The 49 dimension is the 7x7 spatial grid; the 512 dimension is DarkNet channels.
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
    ):
        super().__init__()
        if not 4 <= num_region_queries <= 8:
            raise ValueError("num_region_queries must be in the range [4, 8].")

        self.visual_dim = visual_dim
        self.embed_dim = embed_dim
        self.num_region_queries = num_region_queries
        self.instruction_dim = instruction_dim

        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.region_queries = nn.Parameter(torch.randn(num_region_queries, embed_dim) * 0.02)
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

    def forward(self, visual_tokens, instruction_embedding=None):
        single_frame = visual_tokens.dim() == 3
        if single_frame:
            visual_tokens = visual_tokens.unsqueeze(1)
        if visual_tokens.dim() != 4:
            raise ValueError(
                "visual_tokens must be [B, T, 512, 49] or [B, 512, 49], "
                f"got {tuple(visual_tokens.shape)}."
            )

        batch_size, time_steps, channels, grid_size = visual_tokens.shape
        if channels != self.visual_dim:
            raise ValueError(f"Expected visual channel dim {self.visual_dim}, got {channels}.")

        # [B, T, 512, 49] -> [B*T, 49, 512] -> [B*T, 49, 768]
        visual_seq = visual_tokens.permute(0, 1, 3, 2).reshape(
            batch_size * time_steps,
            grid_size,
            channels,
        )
        visual_seq = self.visual_proj(visual_seq)

        # Learned region prompts query the 49 projected visual grid tokens.
        queries = self.region_queries.unsqueeze(0).expand(batch_size * time_steps, -1, -1)
        instruction = self._project_instruction(instruction_embedding, repeat_factor=time_steps)
        if instruction is not None:
            queries = queries + instruction.unsqueeze(1)

        region_tokens, _ = self.cross_attention(
            query=queries,
            key=visual_seq,
            value=visual_seq,
            need_weights=False,
        )

        if instruction is not None:
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
