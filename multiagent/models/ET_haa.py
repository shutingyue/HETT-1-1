import torch
import math
from collections import defaultdict
from .enc_vl import EncoderVL
from torch import nn
from torch.nn import functional as F

from .goal_predictor import MapEncoder
from .topo_memory import TopoMemoryBuilder


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

        # self.c = nn.Sequential(
        # nn.Linear(768, 256),
        # # nn.BatchNorm1d(64, eps=1e-12),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(256, 32),
        # # nn.BatchNorm1d(64, eps=1e-12),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(32, 4),
        # # nn.BatchNorm1d(768, eps=1e-12),
        # nn.ReLU())

    def forward(self, h, context, mask=None):  # context will be weighted and concat with h
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax 
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        lang_embeds = torch.cat((weighted_context, h), 1)

        lang_embeds = self.tanh(self.linear_out(lang_embeds))
        return lang_embeds, attn


class ET(nn.Module):
    def __init__(self, args):
        """
        transformer agent
        """
        super().__init__()
        self.args = args
        # encoder and visual embeddings
        self.map_encoder = MapEncoder(240)
        self.encoder_vl = EncoderVL(args)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        self.centroid_encoder = nn.Sequential(
            nn.Linear(2, self.args.demb),
            nn.LayerNorm(self.args.demb, eps=1e-12)
        )
        # # feature embeddings
        # self.vis_feat = FeatureFlat(input_shape=self.visual_tensor_shape, output_size=args.demb)
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None

        # self.vis_feat = FeatureFlat(input_shape=(650,7,7), output_size=args.demb)

        self.args = args

        # XVIEW
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(self.args.demb, 256),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
            nn.Tanh()
        )
        self.attention_layer_vision = SoftDotAttention(49)
        self.region_prompt_adapter = None
        self.stop_visual_context_adapter = None
        self.stop_contrast_loss = None
        if (
            bool(getattr(self.args, "use_region_prompt", False))
            and getattr(self.args, "region_prompt_mode", "residual") != "original"
        ):
            from .region_prompt import RegionPromptAdapter

            region_prompt_max_spatial_tokens = int(
                getattr(self.args, "region_prompt_max_spatial_tokens", 0)
            )
            if region_prompt_max_spatial_tokens <= 0:
                region_prompt_max_spatial_tokens = 49
            self.region_prompt_adapter = RegionPromptAdapter(
                visual_dim=512,
                embed_dim=self.args.demb,
                num_region_queries=getattr(self.args, "region_prompt_num", 4),
                num_heads=self.args.encoder_heads,
                dropout=getattr(self.args, "region_prompt_dropout", 0.1),
                instruction_dim=49,
                condition_generation=getattr(self.args, "region_prompt_condition_generation", False),
                fuse_instruction=getattr(self.args, "region_prompt_fuse_instruction", False),
                query_init=getattr(self.args, "region_prompt_query_init", "random"),
                query_scale=getattr(self.args, "region_prompt_query_scale", 0.1),
                use_pos_embed=getattr(self.args, "region_prompt_use_pos_embed", False),
                max_spatial_tokens=region_prompt_max_spatial_tokens,
                attn_topk=getattr(self.args, "region_attn_topk", 5),
            )
        stop_contrast_source = getattr(self.args, "stop_contrast_visual_source", "none")
        stop_contrast_needs_visual_context = (
            bool(getattr(self.args, "use_stop_contrast", False))
            and stop_contrast_source in ("global_attn", "fixed_partition")
        )
        if bool(getattr(self.args, "use_stop_visual_context", False)) or stop_contrast_needs_visual_context:
            from .visual_context import StopVisualContextAdapter

            stop_visual_context_mode = (
                stop_contrast_source if stop_contrast_needs_visual_context
                else getattr(self.args, "stop_visual_context_mode", "global_attn")
            )
            self.stop_visual_context_adapter = StopVisualContextAdapter(
                mode=stop_visual_context_mode,
                visual_dim=512,
                embed_dim=getattr(self.args, "stop_visual_context_dim", self.args.demb),
                num_heads=self.args.encoder_heads,
                dropout=getattr(self.args, "stop_visual_context_dropout", 0.1),
                instruction_dim=49,
                num_regions=getattr(self.args, "stop_visual_context_num_regions", 4),
                topk=getattr(self.args, "stop_visual_context_topk", 5),
            )
        if bool(getattr(self.args, "use_stop_contrast", False)):
            from .stop_contrast import StopContrastLoss

            if stop_contrast_source == "region_prompt" and self.region_prompt_adapter is None:
                raise ValueError(
                    "StopContrast source=region_prompt requires --use_region_prompt "
                    "with region_prompt_mode residual or replace."
                )
            self.stop_contrast_loss = StopContrastLoss(
                visual_source=stop_contrast_source,
                hidden_dim=self.args.demb,
                visual_dim=self.args.demb,
                instruction_dim=49,
                proj_dim=getattr(self.args, "stop_contrast_proj_dim", 256),
                temperature=getattr(self.args, "stop_contrast_temperature", 0.07),
                dropout=getattr(self.args, "dropout_emb", 0.1),
                require_both_pos_neg=getattr(self.args, "stop_contrast_require_both_pos_neg", True),
            )
        self.decoder_2_progress_full = nn.Sequential(
            nn.Linear(self.args.demb, 256),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        self.decoder_2_logits_full = nn.Sequential(
            nn.Linear(self.args.demb, self.args.demb // 2),
            nn.ReLU(),
            nn.Linear(self.args.demb // 2, 1),
        )
        self.decoder_2_goal_full = nn.Sequential(
            nn.Linear(self.args.demb, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )
        self.direction_embedding = nn.Linear(4, self.args.demb)

        self.fc2 = nn.Linear(49, self.args.demb)

        self.fc_map = nn.Linear(self.map_encoder.out_features, args.demb)

        self.text_proj = nn.Linear(768, 768)
        self.grid_proj = nn.Linear(768, 768)
        self.use_topo_memory = args.enable_topo_memory
        self.use_time_decay = bool(getattr(args, "use_time_decay", False))
        if self.use_topo_memory:
            self.topo_memory_builder = TopoMemoryBuilder(args)
            self.topo_goal_offset = nn.Sequential(
                nn.Linear(self.args.demb, self.args.demb),
                nn.ReLU(),
                nn.Linear(self.args.demb, 2),
            )
            self.topo_patch_proj = nn.Linear(self.args.demb, self.args.demb)
            self.topo_visual_gate = nn.Linear(self.args.demb * 2, self.args.demb)
            self.topo_action_gate = nn.Linear(self.args.demb * 2, self.args.demb)
        # Only create the learned decay parameter when the explicit flag is enabled.
        # When disabled, the model has no time-decay state and all history scoring stays
        # on the original similarity-only path.
        if self.use_time_decay:
            self.decay_rate = nn.Parameter(torch.tensor(0.1))
        self.print_counter = 0
        self.topo_debug_cache = None
        self.topo_eval_debug_batches = 0

    def _compute_history_scores(self, tmp_fts, text_ft, time_steps, current_t):
        if tmp_fts.shape[0] > 0:
            grid_fts_weight, _ = (tmp_fts @ text_ft).max(dim=-1)
        else:
            grid_fts_weight = tmp_fts.new_zeros((0,))

        if self.use_time_decay and time_steps is not None and current_t is not None and tmp_fts.shape[0] > 0:
            delta_t = current_t - time_steps
            positive_decay_rate = F.softplus(self.decay_rate)
            time_penalty = positive_decay_rate * delta_t
            grid_fts_weight = grid_fts_weight - time_penalty
        return grid_fts_weight

    def _fuse_patch_context(self, base_feat, patch_feat, gate_layer):
        gate = torch.sigmoid(gate_layer(torch.cat((base_feat, patch_feat), dim=-1)))
        return base_feat + gate * patch_feat

    def _scale_region_context(self, region_context, original_emb_frames):
        scale_mode = getattr(self.args, "region_prompt_scale_mode", "sqrt_dim")
        eps = 1e-6
        if scale_mode == "none":
            return region_context
        if scale_mode == "sqrt_dim":
            return region_context / math.sqrt(region_context.size(-1))
        if scale_mode == "match_original":
            region_norm = torch.norm(region_context, dim=-1, keepdim=True)
            original_norm = torch.norm(original_emb_frames.detach(), dim=-1, keepdim=True)
            return region_context / region_norm.clamp_min(eps) * original_norm
        raise ValueError(f"Unsupported region_prompt_scale_mode: {scale_mode}")

    def _append_region_prompt_diagnostics(
        self,
        stats,
        original_emb_frames,
        region_tokens_all,
        region_context,
        region_attn,
    ):
        eps = 1e-6
        with torch.no_grad():
            original = original_emb_frames.detach().to(torch.float32)
            context = region_context.detach().to(torch.float32)
            tokens = region_tokens_all.detach().to(torch.float32)

            original_norm = torch.nan_to_num(original.norm(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            context_norm = torch.nan_to_num(context.norm(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            residual_ratio = (
                float(getattr(self.args, "region_prompt_alpha", 0.0))
                * context_norm
                / original_norm.clamp_min(eps)
            )

            stats["region_context_norm"].append(float(context_norm.mean().item()))
            stats["original_emb_norm"].append(float(original_norm.mean().item()))
            stats["region_residual_ratio"].append(
                float(torch.nan_to_num(residual_ratio, nan=0.0, posinf=0.0, neginf=0.0).mean().item())
            )

            if self.region_prompt_adapter is not None:
                query_stats = self.region_prompt_adapter.region_query_diagnostics()
                for key in (
                    "region_query_norm",
                    "region_query_offdiag_cos",
                    "spatial_pos_embed_norm",
                    "spatial_pos_embed_offdiag_cos",
                ):
                    value = query_stats.get(key)
                    if value is not None and math.isfinite(value):
                        stats[key].append(value)

            attn = region_attn.detach().to(torch.float32)
            if attn.dim() < 2:
                attn = attn.reshape(1, -1)
            p = attn.reshape(-1, attn.shape[-1])
            if p.shape[-1] > 0:
                row_sum = p.sum(dim=-1, keepdim=True)
                looks_like_prob = (
                    bool(torch.isfinite(p).all().item())
                    and float(p.min().item()) >= -eps
                    and bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3, rtol=1e-3))
                )
                if not looks_like_prob:
                    p = torch.softmax(p, dim=-1)
                p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
                num_regions = p.shape[-1]
                entropy = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
                if num_regions > 1:
                    entropy = entropy / math.log(num_regions)
                else:
                    entropy = entropy.new_zeros(entropy.shape)
                stats["region_attn_entropy"].append(float(entropy.mean().item()))
                stats["region_attn_max"].append(float(p.max(dim=-1).values.mean().item()))

            if tokens.dim() == 4 and tokens.shape[-2] > 1:
                region_norm = F.normalize(tokens, dim=-1, eps=eps)
                sim = torch.matmul(region_norm, region_norm.transpose(-1, -2))
                num_regions = sim.shape[-1]
                offdiag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=sim.device)
                offdiag_sim = sim[..., offdiag_mask]
                stats["region_token_offdiag_cos"].append(
                    float(torch.nan_to_num(offdiag_sim, nan=0.0, posinf=0.0, neginf=0.0).mean().item())
                )

    def _append_topo_token_debug_stats(
        self,
        stats_accumulator,
        global_tokens,
        global_padding_mask,
        local_tokens=None,
        local_padding_mask=None,
    ):
        """Collect lightweight global/local topo token stats for training logs."""

        global_valid = ~global_padding_mask.bool()
        global_counts = global_valid.sum(dim=1).to(torch.float32)
        stats_accumulator["global_token_count"].append(float(global_counts.mean().item()))
        global_norms = global_tokens.norm(dim=-1).masked_select(global_valid)
        if global_norms.numel() > 0:
            stats_accumulator["global_token_norm_mean"].append(float(global_norms.mean().item()))
        else:
            stats_accumulator["global_token_norm_mean"].append(0.0)

        local_norms = global_tokens.new_zeros((0,))
        if local_tokens is not None:
            if local_padding_mask is None:
                local_valid = torch.ones(local_tokens.shape[:2], device=local_tokens.device, dtype=torch.bool)
            else:
                local_valid = ~local_padding_mask.bool()
            local_counts = local_valid.sum(dim=1).to(torch.float32)
            stats_accumulator["local_token_count"].append(float(local_counts.mean().item()))
            local_norms = local_tokens.norm(dim=-1).masked_select(local_valid)
            if local_norms.numel() > 0:
                stats_accumulator["local_token_norm_mean"].append(float(local_norms.mean().item()))
            else:
                stats_accumulator["local_token_norm_mean"].append(0.0)
        else:
            stats_accumulator["local_token_count"].append(0.0)
            stats_accumulator["local_token_norm_mean"].append(0.0)

        if global_norms.numel() > 0 and local_norms.numel() > 0:
            topo_norms = torch.cat((global_norms, local_norms), dim=0)
        elif global_norms.numel() > 0:
            topo_norms = global_norms
        else:
            topo_norms = local_norms
        if topo_norms.numel() > 0:
            stats_accumulator["topo_token_norm_mean"].append(float(topo_norms.mean().item()))
            stats_accumulator["topo_token_norm_std"].append(
                float(topo_norms.std(unbiased=False).item()) if topo_norms.numel() > 1 else 0.0
            )
        else:
            stats_accumulator["topo_token_norm_mean"].append(0.0)
            stats_accumulator["topo_token_norm_std"].append(0.0)

    def _validate_topo_history_alignment(
        self, batch_idx, history_features, history_index, history_times, history_xy=None
    ):
        feature_len = int(history_features.shape[0])
        index_len = int(history_index.shape[0])
        time_len = int(history_times.shape[0])
        xy_len = feature_len if history_xy is None else int(history_xy.shape[0])
        if feature_len != index_len or feature_len != time_len or feature_len != xy_len:
            raise RuntimeError(
                "Topo history alignment mismatch at batch {}: history_features={}, history_cell_ids={}, history_times={}, history_xy={}".format(
                    batch_idx, feature_len, index_len, time_len, xy_len
                )
            )

    def _compress_spatial_tokens(self, cell_features, cell_positions, current_grid):
        """
        Build the effective spatial token list that enters the transformer.
        `current_grid` and `grid_index` both use the flattened cell index
        row_id * grid_size + col_id. This is the same coordinate convention as env.py,
        so the near/far split is computed in exactly the same grid space as the history.

        Near cells stay as fine tokens. Far cells are grouped into coarse regions and
        summarized into fewer tokens. `cell_to_token` stores, for each original cell id
        in [0, grid_size^2), which compressed token now represents that cell.
        """
        num_cells = cell_features.shape[0]
        if not self.args.spatial_compression:
            cell_to_token = torch.arange(num_cells, device=cell_features.device, dtype=torch.long)
            stats = {
                'tokens_before': float(num_cells),
                'tokens_after': float(num_cells),
                'near_tokens': float(num_cells),
                'far_summary_tokens': 0.0,
                'pruned_tokens': 0.0,
                'merged_away_tokens': 0.0,
            }
            return cell_features, cell_positions, cell_to_token, stats

        grid_size = self.args.grid_size
        current_grid = int(current_grid.item())
        current_row = current_grid // grid_size
        current_col = current_grid % grid_size

        device = cell_features.device
        rows = torch.arange(grid_size, device=device).unsqueeze(1).repeat(1, grid_size).reshape(-1)
        cols = torch.arange(grid_size, device=device).unsqueeze(0).repeat(grid_size, 1).reshape(-1)
        distance = torch.abs(rows - current_row) + torch.abs(cols - current_col)
        near_mask = distance <= self.args.spatial_dist_threshold

        token_features = []
        token_positions = []
        cell_to_token = torch.empty(num_cells, device=device, dtype=torch.long)

        near_indices = torch.nonzero(near_mask, as_tuple=False).flatten()
        for idx in near_indices.tolist():
            token_idx = len(token_features)
            token_features.append(cell_features[idx])
            token_positions.append(cell_positions[idx])
            cell_to_token[idx] = token_idx

        far_indices = torch.nonzero(~near_mask, as_tuple=False).flatten()
        far_groups = {}
        coarse_size = max(int(self.args.spatial_far_coarse_size), 1)
        for idx in far_indices.tolist():
            key = (int(rows[idx].item()) // coarse_size, int(cols[idx].item()) // coarse_size)
            far_groups.setdefault(key, []).append(idx)

        for _, members in sorted(far_groups.items()):
            token_idx = len(token_features)
            member_index = torch.tensor(members, device=device, dtype=torch.long)
            token_features.append(cell_features.index_select(0, member_index).mean(dim=0))
            token_positions.append(cell_positions.index_select(0, member_index).mean(dim=0))
            cell_to_token[member_index] = token_idx

        compressed_features = torch.stack(token_features, dim=0)
        compressed_positions = torch.stack(token_positions, dim=0)
        far_summary_tokens = len(far_groups)
        tokens_after = compressed_features.shape[0]
        stats = {
            'tokens_before': float(num_cells),
            'tokens_after': float(tokens_after),
            'near_tokens': float(near_indices.numel()),
            'far_summary_tokens': float(far_summary_tokens),
            'pruned_tokens': 0.0,
            'merged_away_tokens': float(num_cells - tokens_after),
        }
        return compressed_features, compressed_positions, cell_to_token, stats

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        emb_lang = inputs["lang"]

        map_feat = self.map_encoder(inputs['maps'])
        # print(torch.isnan(map_feat).any(), torch.isinf(map_feat).any())

        # # embed frames and direiction (650,49) --> 768
        # im_feature = inputs["frames"]
        # embed_frame, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:,-1, :, :])
        # h_sali = self.fc(embed_frame).view(-1,1,8,8)
        # pred_saliency = nn.functional.interpolate(h_sali,size=(224,224),mode='bilinear',align_corners=False)
        # frames_pad_emb = self.vis_feat(im_feature.view(-1, 650,7,7)).view(*im_feature.shape[:2], -1)

        # embed frames and direiction:
        # im_feature: [B, T, 512, N] for adapters; the legacy attention path below
        # still summarizes each frame to a [B, T, 49] feature for fc2.
        # original_att_frame_feature: [B, T, 49]
        # original_emb_frames: [B, T, 768], the unchanged HETT visual frame path.
        im_feature = inputs["frames"]
        original_att_frame_feature = torch.zeros((im_feature.shape[0], 0, 49), device=im_feature.device)
        for i in range(im_feature.shape[1]):
            att_single_frame_feature, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:, i, :, :])
            original_att_frame_feature = torch.concat(
                (original_att_frame_feature, att_single_frame_feature.unsqueeze(1)),
                axis=1,
            )

        original_emb_frames = self.fc2(original_att_frame_feature.view(-1, 49)).view(*im_feature.shape[:2], -1)
        emb_frames = original_emb_frames
        compression_stats = defaultdict(list)
        stop_visual_context = None
        region_context = None

        use_stop_contrast = bool(getattr(self.args, "use_stop_contrast", False))
        stop_contrast_source = getattr(self.args, "stop_contrast_visual_source", "none")
        stop_contrast_needs_visual_context = (
            use_stop_contrast
            and stop_contrast_source in ("global_attn", "fixed_partition")
        )
        use_stop_visual_context = bool(getattr(self.args, "use_stop_visual_context", False))
        if (
            self.stop_visual_context_adapter is not None
            and (use_stop_visual_context or stop_contrast_needs_visual_context)
        ):
            stop_visual_context, stop_visual_context_diagnostics = self.stop_visual_context_adapter(
                im_feature,
                inputs["lang_cls"],
            )
            for key, value in stop_visual_context_diagnostics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    compression_stats[key].append(float(value))

        use_region_prompt = (
            bool(getattr(self.args, "use_region_prompt", False))
            and getattr(self.args, "region_prompt_mode", "residual") != "original"
            and self.region_prompt_adapter is not None
        )
        if use_region_prompt:
            use_region_attn_diversity = (
                self.training
                and bool(getattr(self.args, "use_region_attn_diversity", False))
            )
            region_tokens_all = torch.zeros(
                (
                    im_feature.shape[0],
                    0,
                    self.region_prompt_adapter.num_region_queries,
                    self.args.demb,
                ),
                device=im_feature.device,
            )
            visual_token_offdiag_values = []
            raw_visual_token_offdiag_values = []
            region_input_values = defaultdict(list)
            region_gen_attn_values = defaultdict(list)
            projected_query_values = defaultdict(list)
            query_spatial_affinity_values = defaultdict(list)
            region_attn_diversity_losses = []
            for i in range(im_feature.shape[1]):
                region_tokens = self.region_prompt_adapter(
                    im_feature[:, i, :, :],
                    inputs["lang_cls"],
                    compute_attention_diversity=use_region_attn_diversity,
                    attention_diversity_mode=getattr(
                        self.args,
                        "region_attn_diversity_mode",
                        "cosine_square",
                    ),
                )
                region_tokens_all = torch.concat((region_tokens_all, region_tokens.unsqueeze(1)), axis=1)
                if use_region_attn_diversity:
                    region_attn_diversity_loss = getattr(
                        self.region_prompt_adapter,
                        "latest_region_attn_diversity_loss",
                        None,
                    )
                    if region_attn_diversity_loss is not None:
                        region_attn_diversity_losses.append(region_attn_diversity_loss)
                visual_token_offdiag_cos = getattr(
                    self.region_prompt_adapter,
                    "latest_visual_token_offdiag_cos",
                    None,
                )
                raw_visual_token_offdiag_cos = getattr(
                    self.region_prompt_adapter,
                    "latest_raw_visual_token_offdiag_cos",
                    None,
                )
                if visual_token_offdiag_cos is not None and math.isfinite(visual_token_offdiag_cos):
                    visual_token_offdiag_values.append(visual_token_offdiag_cos)
                if raw_visual_token_offdiag_cos is not None and math.isfinite(raw_visual_token_offdiag_cos):
                    raw_visual_token_offdiag_values.append(raw_visual_token_offdiag_cos)
                for key, value in getattr(
                    self.region_prompt_adapter,
                    "latest_input_diagnostics",
                    {},
                ).items():
                    if value is not None and math.isfinite(value):
                        region_input_values[key].append(value)
                for key, value in getattr(
                    self.region_prompt_adapter,
                    "latest_generation_attention_diagnostics",
                    {},
                ).items():
                    if value is not None and math.isfinite(value):
                        region_gen_attn_values[key].append(value)
                for key, value in getattr(
                    self.region_prompt_adapter,
                    "latest_projected_query_diagnostics",
                    {},
                ).items():
                    if value is not None and math.isfinite(value):
                        projected_query_values[key].append(value)
                for key, value in getattr(
                    self.region_prompt_adapter,
                    "latest_query_spatial_affinity_diagnostics",
                    {},
                ).items():
                    if value is not None and math.isfinite(value):
                        query_spatial_affinity_values[key].append(value)

            if visual_token_offdiag_values:
                compression_stats["visual_token_offdiag_cos"].append(
                    sum(visual_token_offdiag_values) / len(visual_token_offdiag_values)
                )
            if raw_visual_token_offdiag_values:
                compression_stats["raw_visual_token_offdiag_cos"].append(
                    sum(raw_visual_token_offdiag_values) / len(raw_visual_token_offdiag_values)
                )
            for key, values in region_input_values.items():
                if values:
                    compression_stats[key].append(sum(values) / len(values))
            for key, values in region_gen_attn_values.items():
                if values:
                    compression_stats[key].append(sum(values) / len(values))
            for key, values in projected_query_values.items():
                if values:
                    compression_stats[key].append(sum(values) / len(values))
            for key, values in query_spatial_affinity_values.items():
                if values:
                    compression_stats[key].append(sum(values) / len(values))
            if use_region_attn_diversity:
                if region_attn_diversity_losses:
                    compression_stats["region_attn_diversity_loss"].append(
                        torch.stack([loss.reshape(()) for loss in region_attn_diversity_losses]).mean()
                    )
                else:
                    compression_stats["region_attn_diversity_loss"].append(im_feature.sum() * 0.0)

            region_context, region_attn = self.region_prompt_adapter.select_region_context(
                region_tokens_all,
                inputs["lang_cls"],
            )
            region_context_scaled = self._scale_region_context(region_context, original_emb_frames)
            if self.args.region_prompt_mode == "replace":
                emb_frames = region_context_scaled
            elif self.args.region_prompt_mode == "residual":
                emb_frames = original_emb_frames + self.args.region_prompt_alpha * region_context_scaled
            else:
                emb_frames = original_emb_frames
            self._append_region_prompt_diagnostics(
                compression_stats,
                original_emb_frames,
                region_tokens_all,
                region_context_scaled,
                region_attn,
            )

        emb_maps = self.fc_map(map_feat).view(im_feature.shape[0], -1, 768)
        # print('sss', emb_frames.shape, emb_maps.shape)
        # print(map_feat.shape)

        emb_directions = self.direction_embedding(inputs["directions"].view(-1, 4)).view(im_feature.shape[0], -1,
                                                                                         768)  # (batch, embedding_size)
        stop_contrast_score = None
        if self.training and use_stop_contrast and self.stop_contrast_loss is not None:
            stop_contrast_visual_context = None
            if stop_contrast_source in ("global_attn", "fixed_partition"):
                stop_contrast_visual_context = stop_visual_context
            elif stop_contrast_source == "region_prompt":
                stop_contrast_visual_context = region_context
            action_hidden = emb_frames + emb_directions
            stop_contrast_score = self.stop_contrast_loss.score(
                action_hidden=action_hidden,
                instruction=emb_lang[:, 0, :],
                visual_context=stop_contrast_visual_context,
                detach_visual=bool(getattr(self.args, "stop_contrast_detach_visual", False)),
            )
        batch_size = emb_lang.shape[0]

        text_fts = self.text_proj(emb_lang).permute(0, 2, 1)
        grid_fts = inputs['grid_fts']
        grid_map_indexs = inputs['grid_index']
        current_grids = inputs['current_grid']
        topo_memory_outputs = inputs.get('topo_memory_outputs', None)
        # 【新增】：从 inputs 字典中安全地取出时间参数
        time_steps = inputs.get('time_steps', None)
        current_t = inputs.get('current_t', None)
        topo_xy_history = inputs.get('topo_xy', None)
        topo_landmarks = inputs.get('topo_landmarks', None)
        topo_eval_debug = False
        topo_eval_batch_idx = None

        candidate_token_features = []
        candidate_token_positions = []
        cell_to_token_maps = []
        base_positions = inputs['candidates'].to(emb_lang.device).to(torch.float32)
        grid_cell_count = self.args.grid_size ** 2
        positive_decay_rate = F.softplus(self.decay_rate) if self.use_time_decay else None

        topo_history_features = []
        topo_history_index = []
        topo_history_times = []
        topo_history_xy = []
        topo_fallback_features = []

        for b in range(batch_size):
            tmp_fts = grid_fts[b].to(torch.float32)
            b_time_steps = time_steps[b] if time_steps is not None else None
            grid_fts_weight = self._compute_history_scores(tmp_fts, text_fts[b], b_time_steps, current_t)
            tmp_fts = self.grid_proj(tmp_fts)
            cell_features = torch.zeros(grid_cell_count, self.args.demb, device=emb_lang.device, dtype=tmp_fts.dtype)
            for i in range(grid_cell_count):
                cell_fts = tmp_fts[grid_map_indexs[b] == i]
                if cell_fts.shape[0] == 0:
                    continue
                cell_scores = grid_fts_weight[grid_map_indexs[b] == i]
                cell_features[i] = (
                    cell_fts * torch.softmax(cell_scores, dim=-1).unsqueeze(-1)
                ).sum(dim=0)

            if self.use_topo_memory:
                topo_history_features.append(tmp_fts)
                topo_history_index.append(grid_map_indexs[b].long())
                topo_history_times.append(
                    b_time_steps if b_time_steps is not None else tmp_fts.new_zeros(tmp_fts.shape[0])
                )
                if topo_xy_history is not None:
                    topo_history_xy.append(topo_xy_history[b].to(torch.float32))
                else:
                    topo_history_xy.append(base_positions[b].new_zeros((tmp_fts.shape[0], 2)))
                topo_fallback_features.append(cell_features[current_grids[b].long()])
            else:
                compressed_features, compressed_positions, cell_to_token, stats = self._compress_spatial_tokens(
                    cell_features,
                    base_positions[b],
                    current_grids[b],
                )
                candidate_token_features.append(compressed_features)
                candidate_token_positions.append(compressed_positions)
                cell_to_token_maps.append(cell_to_token)
                for key, value in stats.items():
                    compression_stats[key].append(value)

        topo_outputs = None
        if self.use_topo_memory and topo_memory_outputs is not None:
            topo_outputs = topo_memory_outputs
            # Global retrieved place tokens remain the spatial candidates for the coarse
            # target head in the current minimal path.
            # TODO: route global tokens to coarse stage explicitly if the model is split.
            global_retrieved_tokens = topo_outputs.get(
                "global_retrieved_tokens_padded",
                topo_outputs["node_features_padded"],
            )
            local_subgraph_tokens = topo_outputs.get("local_subgraph_tokens_padded", None)
            emb_candidates = torch.zeros(
                batch_size,
                global_retrieved_tokens.shape[1],
                self.args.demb,
                device=emb_lang.device,
            )
            spatial_padding_mask = topo_outputs["node_padding_mask"]
            cell_to_token_map = topo_outputs["cell_to_node_map"]
            for b in range(batch_size):
                token_count = int((~spatial_padding_mask[b]).sum().item())
                if token_count == 0:
                    continue
                pos_embed = (
                    self.candidate_encoder(topo_outputs["node_positions_padded"][b, :token_count])
                    * emb_lang[b:b + 1, :1, :]
                )
                emb_candidates[b, :token_count] = (
                    pos_embed.squeeze(0) + global_retrieved_tokens[b, :token_count]
                )
            for key, value in topo_outputs["stats"].items():
                compression_stats[key].append(value)
            self._append_topo_token_debug_stats(
                compression_stats,
                global_retrieved_tokens,
                spatial_padding_mask,
                local_subgraph_tokens,
                topo_outputs.get("local_token_padding_mask", None),
            )
        elif self.use_topo_memory and self.args.topo_rebuild_fallback:
            topo_eval_debug = (
                not self.training and self.use_topo_memory and self.topo_eval_debug_batches < 3
            )
            topo_eval_batch_idx = self.topo_eval_debug_batches
            history_len = grid_fts.shape[1]
            if history_len == 0:
                topo_history_features_tensor = grid_fts.new_zeros(batch_size, 0, self.args.demb).to(torch.float32)
                topo_history_index_tensor = grid_map_indexs.new_zeros(batch_size, 0, dtype=torch.long)
                topo_history_times_tensor = grid_fts.new_zeros(batch_size, 0).to(torch.float32)
                topo_history_xy_tensor = base_positions.new_zeros(batch_size, 0, 2).to(torch.float32)
            else:
                for b in range(batch_size):
                    self._validate_topo_history_alignment(
                        b,
                        topo_history_features[b],
                        topo_history_index[b],
                        topo_history_times[b],
                        topo_history_xy[b],
                    )
                topo_history_features_tensor = torch.stack(topo_history_features, dim=0)
                topo_history_index_tensor = torch.stack(topo_history_index, dim=0)
                topo_history_times_tensor = torch.stack(topo_history_times, dim=0).to(torch.float32)
                topo_history_xy_tensor = torch.stack(topo_history_xy, dim=0).to(torch.float32)
            topo_fallback_tensor = torch.stack(topo_fallback_features, dim=0)
            if topo_eval_debug:
                history_features_len = [int(x.shape[0]) for x in topo_history_features]
                history_cell_ids_len = [int(x.shape[0]) for x in topo_history_index]
                history_times_len = [int(x.shape[0]) for x in topo_history_times]
                history_xy_len = [int(x.shape[0]) for x in topo_history_xy]
                print(
                    '[TopoEvalHistory] batch={} history_features_len={} history_cell_ids_len={} history_times_len={} history_xy_len={}'.format(
                        topo_eval_batch_idx,
                        history_features_len,
                        history_cell_ids_len,
                        history_times_len,
                        history_xy_len,
                    )
                )
            topo_outputs = self.topo_memory_builder(
                topo_history_features_tensor,
                topo_history_index_tensor,
                topo_history_times_tensor,
                topo_history_xy_tensor,
                topo_landmarks,
                base_positions,
                # Compatibility-first fallback: use the first language token as a pooled goal embedding.
                # Future work can replace this with a better lang_cls / instruction pooling path.
                emb_lang[:, 0, :],
                current_grids,
                topo_fallback_tensor,
                positive_decay_rate,
            )
            # Global tokens are still used as coarse target candidates; local tokens are
            # summarized later into the fine action/progress refinement path.
            # TODO: route local tokens to fine stage explicitly if the transformer is split.
            global_retrieved_tokens = topo_outputs.get(
                "global_retrieved_tokens_padded",
                topo_outputs["node_features_padded"],
            )
            local_subgraph_tokens = topo_outputs.get("local_subgraph_tokens_padded", None)
            emb_candidates = torch.zeros(
                batch_size,
                global_retrieved_tokens.shape[1],
                self.args.demb,
                device=emb_lang.device,
            )
            spatial_padding_mask = topo_outputs["node_padding_mask"]
            cell_to_token_map = topo_outputs["cell_to_node_map"]
            for b in range(batch_size):
                token_count = int((~spatial_padding_mask[b]).sum().item())
                if token_count == 0:
                    continue
                pos_embed = (
                    self.candidate_encoder(topo_outputs["node_positions_padded"][b, :token_count])
                    * emb_lang[b:b + 1, :1, :]
                )
                emb_candidates[b, :token_count] = (
                    pos_embed.squeeze(0) + global_retrieved_tokens[b, :token_count]
                )
            for key, value in topo_outputs["stats"].items():
                compression_stats[key].append(value)
            self._append_topo_token_debug_stats(
                compression_stats,
                global_retrieved_tokens,
                spatial_padding_mask,
                local_subgraph_tokens,
                topo_outputs.get("local_token_padding_mask", None),
            )
        else:
            max_cell_num = max(token.shape[0] for token in candidate_token_features)
            emb_candidates = torch.zeros(batch_size, max_cell_num, self.args.demb, device=emb_lang.device)
            # True means this slot is only batch padding and should be ignored by the transformer.
            # False means this compressed spatial token is real and should participate in attention.
            spatial_padding_mask = torch.ones(batch_size, max_cell_num, device=emb_lang.device, dtype=torch.bool)
            # Each original cell points to the compressed token that represents it.
            # Near cells map 1:1; far cells in the same coarse region share one token index.
            cell_to_token_map = torch.zeros(batch_size, grid_cell_count, device=emb_lang.device, dtype=torch.long)

            for b in range(batch_size):
                token_count = candidate_token_features[b].shape[0]
                spatial_padding_mask[b, :token_count] = False
                cell_to_token_map[b] = cell_to_token_maps[b]
                pos_embed = self.candidate_encoder(candidate_token_positions[b]) * emb_lang[b:b + 1, :1, :]
                emb_candidates[b, :token_count] = pos_embed.squeeze(0) + candidate_token_features[b]

        max_cell_num = emb_candidates.shape[1]

        # emb_centroids = (self.centroid_encoder(inputs['centroids']) * emb_lang[:, 0, :]).view(im_feature.shape[0], -1, 768)
        # emb_centroids = self.centroid_encoder(inputs['centroids']).view(im_feature.shape[0], -1, 768)
        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl.forward_with_map(
            emb_lang,
            emb_frames,
            emb_directions,
            emb_maps,
            emb_candidates,
            spatial_padding_mask=spatial_padding_mask,

            # inputs['lenths']
        )

        # use outputs corresponding to last visual frames for prediction only
        encoder_out_visual = encoder_out[:, emb_lang.shape[1]]
        encoder_out_direction = encoder_out[:, emb_lang.shape[1] + 1]
        # encoder_out_candidates = encoder_out[:, emb_lang.shape[1] + 3: emb_lang.shape[1] + 3 + emb_candidates.shape[1]]
        encoder_out_candidates = encoder_out[:, emb_lang.shape[1] + 3:]
        encoder_out_centroids = encoder_out[:, emb_lang.shape[1] + 2]
        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_decoder_input = encoder_out_direction.reshape(-1, self.args.demb)
        goal_decoder_input = encoder_out_centroids.reshape(-1, self.args.demb)
        target_decoder_input = encoder_out_candidates.reshape(-1, max_cell_num, self.args.demb)

        if self.use_topo_memory and topo_outputs is not None:
            # Local active-subgraph summaries are kept out of the global candidate list and
            # fused only into visual/action decoding for fine-stage refinement.
            local_context = topo_outputs["local_context"]
            local_patch_context = topo_outputs["local_patch_context"]
            masked_node_logits = self.decoder_2_logits_full(target_decoder_input).squeeze(-1)
            masked_node_logits = masked_node_logits.masked_fill(spatial_padding_mask, -1e9)
            fully_masked = spatial_padding_mask.all(dim=1)
            if fully_masked.any():
                masked_node_logits = masked_node_logits.clone()
                masked_node_logits[fully_masked, 0] = 0.0
            anchor_probs = torch.softmax(masked_node_logits, dim=1)
            valid_anchor_probs = anchor_probs.masked_fill(spatial_padding_mask, 0.0)
            if fully_masked.any():
                valid_anchor_probs = valid_anchor_probs.clone()
                valid_anchor_probs[fully_masked, 0] = 1.0
            anchor_norm = valid_anchor_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
            valid_anchor_probs = valid_anchor_probs / anchor_norm
            anchor_features = torch.sum(
                valid_anchor_probs.unsqueeze(-1) * target_decoder_input, dim=1
            )
            anchor_centers = torch.sum(
                valid_anchor_probs.unsqueeze(-1) * topo_outputs["node_positions_padded"], dim=1
            )
            offset_xy = self.topo_goal_offset(anchor_features)
            pred_goals = torch.clamp(
                anchor_centers + self.args.topo_offset_scale * torch.tanh(offset_xy),
                min=0.0,
                max=1.0,
            )
            if topo_eval_debug:
                node_positions = topo_outputs["node_positions_padded"].detach().cpu()
                base_positions_cpu = base_positions.detach().cpu()
                anchor_centers_cpu = anchor_centers.detach().cpu()
                offset_xy_cpu = offset_xy.detach().cpu()
                pred_goals_cpu = pred_goals.detach().cpu()
                valid_node_mask_cpu = (~spatial_padding_mask).detach().cpu()
                for b in range(batch_size):
                    valid_count = int(valid_node_mask_cpu[b].sum().item())
                    valid_nodes = node_positions[b, :valid_count]
                    candidate_positions = base_positions_cpu[b]
                    candidate_min = candidate_positions.min(dim=0).values.tolist()
                    candidate_max = candidate_positions.max(dim=0).values.tolist()
                    if valid_count > 0:
                        node_min = valid_nodes.min(dim=0).values.tolist()
                        node_max = valid_nodes.max(dim=0).values.tolist()
                        node_preview = valid_nodes[:3].tolist()
                    else:
                        node_min = ['empty', 'empty']
                        node_max = ['empty', 'empty']
                        node_preview = []
                    pred_goal_sample = pred_goals_cpu[b]
                    inside_unit_box = bool(((pred_goal_sample >= 0.0) & (pred_goal_sample <= 1.0)).all().item())
                    near_edge = bool(((pred_goal_sample <= 0.05) | (pred_goal_sample >= 0.95)).any().item())
                    edge_status = 'edge_near' if near_edge else 'not_edge_near'
                    print(
                        '[TopoEvalSpatial] batch={} sample={} valid_nodes={} '
                        'anchor_centers={} offset={} pred_goals={} '
                        'candidate_range=({}, {}) node_range=({}, {}) '
                        'node_preview={} inside_unit_box={} edge_status={}'.format(
                            topo_eval_batch_idx,
                            b,
                            valid_count,
                            anchor_centers_cpu[b].tolist(),
                            offset_xy_cpu[b].tolist(),
                            pred_goal_sample.tolist(),
                            candidate_min,
                            candidate_max,
                            node_min,
                            node_max,
                            node_preview,
                            inside_unit_box,
                            edge_status,
                        )
                    )
                pred_goals_in_unit = bool(((pred_goals_cpu >= 0.0) & (pred_goals_cpu <= 1.0)).all().item())
                edge_fraction = float(
                    ((pred_goals_cpu <= 0.05) | (pred_goals_cpu >= 0.95)).to(torch.float32).mean().item()
                )
                print(
                    '[TopoEvalSpatialSummary] batch={} pred_goals_range=({}, {}) '
                    'candidate_global_range=({}, {}) pred_goals_in_unit_box={} edge_fraction={:.4f}'.format(
                        topo_eval_batch_idx,
                        pred_goals_cpu.min(dim=0).values.tolist(),
                        pred_goals_cpu.max(dim=0).values.tolist(),
                        base_positions_cpu.amin(dim=(0, 1)).tolist(),
                        base_positions_cpu.amax(dim=(0, 1)).tolist(),
                        pred_goals_in_unit,
                        edge_fraction,
                    )
                )

            anchor_idx = torch.argmax(masked_node_logits, dim=1)
            anchor_patch = topo_outputs["node_patch_padded"][
                torch.arange(batch_size, device=emb_lang.device), anchor_idx
            ]
            neighbor_index = topo_outputs["neighbor_index_padded"][
                torch.arange(batch_size, device=emb_lang.device), anchor_idx
            ]
            neighbor_mask = topo_outputs["neighbor_mask_padded"][
                torch.arange(batch_size, device=emb_lang.device), anchor_idx
            ]
            gathered_neighbors = topo_outputs["node_patch_padded"].gather(
                1,
                neighbor_index.unsqueeze(-1).expand(-1, -1, self.args.demb),
            )
            if gathered_neighbors.shape[1] > 0:
                neighbor_weights = (~neighbor_mask).to(gathered_neighbors.dtype).unsqueeze(-1)
                neighbor_patch = (gathered_neighbors * neighbor_weights).sum(dim=1)
                neighbor_patch = neighbor_patch / neighbor_weights.sum(dim=1).clamp_min(1.0)
            else:
                neighbor_patch = torch.zeros_like(anchor_patch)
            patch_context = self.topo_patch_proj(anchor_patch + neighbor_patch + local_patch_context)
            decoder_input = self._fuse_patch_context(decoder_input, patch_context, self.topo_visual_gate)
            action_decoder_input = self._fuse_patch_context(
                action_decoder_input, patch_context, self.topo_action_gate
            )
            decoder_input = self._fuse_patch_context(decoder_input, local_context, self.topo_visual_gate)
            action_decoder_input = self._fuse_patch_context(
                action_decoder_input, local_context, self.topo_action_gate
            )
            target_logits_token = masked_node_logits
            if topo_eval_debug:
                self.topo_eval_debug_batches += 1
        else:
            pred_goals = self.decoder_2_goal_full(goal_decoder_input)
            target_logits_token = self.decoder_2_logits_full(target_decoder_input).squeeze(-1)

        # decoder_input = emb_directions[:,-1].reshape(-1, self.args.demb)
        output = self.decoder_2_action_full(action_decoder_input)
        norm = torch.norm(output, dim=1, keepdim=True) + 1e-6  # 避免除以零
        direction = output / norm

        progress = self.decoder_2_progress_full(decoder_input)

        # The training label still lives in the original grid_size^2 cell space. We therefore
        # gather each original cell's logit from the compressed token that represents it.
        # If several far cells were merged into one coarse token, those cells intentionally
        # share that summary token's logit so the existing CE supervision stays compatible.
        if self.use_topo_memory:
            token_count = int(target_logits_token.shape[1])
            map_min = int(cell_to_token_map.min().item()) if cell_to_token_map.numel() > 0 else 0
            map_max = int(cell_to_token_map.max().item()) if cell_to_token_map.numel() > 0 else -1
            if map_min < 0 or map_max >= token_count:
                raise RuntimeError(
                    "Invalid topo gather index: cell_to_token_map min={}, max={}, token_count={}".format(
                        map_min, map_max, token_count
                    )
                )
        target_logits = torch.gather(target_logits_token, 1, cell_to_token_map).unsqueeze(-1)
        # print(encoder_out_candidates.shape)

        # print(direction, progress, goal_logits)
        # =================== 【新增：参数监控探头】 ===================
        # self.training 是 PyTorch 自带的属性，确保我们只在训练阶段打印，验证/测试时不打印
        if self.training and self.use_time_decay:
            self.print_counter += 1
            # 每调用 500 次前向传播，打印一次
            if self.print_counter % 500 == 0:
                # F.softplus(self.decay_rate) 是我们实际使用的正数衰减值
                # .detach().item() 是把带着梯度的 Tensor 安全地转换成普通的 Python 浮点数，防止内存泄漏
                current_decay_val = F.softplus(self.decay_rate).detach().item()
                print(f"\n---> [Monitor] Forward Steps: {self.print_counter}, Learned Decay Rate: {current_decay_val:.6f} <---")
        # =============================================================

        mean_stats = {}
        for key, values in compression_stats.items():
            if not values:
                continue
            if any(torch.is_tensor(value) for value in values):
                tensor_values = [
                    value.reshape(()) if torch.is_tensor(value) else im_feature.new_tensor(float(value))
                    for value in values
                ]
                mean_stats[key] = torch.stack(tensor_values).mean()
            else:
                mean_stats[key] = float(sum(values) / max(len(values), 1))
        if stop_visual_context is not None:
            mean_stats["stop_visual_context"] = stop_visual_context
            mean_stats["stop_visual_context_mode"] = getattr(
                self.args,
                "stop_visual_context_mode",
                "global_attn",
            )
        if region_context is not None:
            mean_stats["region_context"] = region_context
        if stop_contrast_score is not None:
            mean_stats["stop_contrast_score"] = stop_contrast_score
        if self.use_topo_memory and topo_outputs is not None:
            mean_stats['valid_topo_nodes'] = float((~spatial_padding_mask).sum(dim=1).to(torch.float32).mean().item())
            mean_stats['max_cell_to_token'] = float(cell_to_token_map.max().item()) if cell_to_token_map.numel() > 0 else -1.0
            if not self.training:
                self.topo_debug_cache = {
                    'pred_goals_min': float(pred_goals.min().item()),
                    'pred_goals_max': float(pred_goals.max().item()),
                    'pred_goals_mean': float(pred_goals.mean().item()),
                    'valid_topo_nodes': mean_stats['valid_topo_nodes'],
                    'max_cell_to_token': mean_stats['max_cell_to_token'],
                    'target_logits_token_shape': tuple(target_logits_token.shape),
                    'cell_to_token_map_min': int(cell_to_token_map.min().item()) if cell_to_token_map.numel() > 0 else 0,
                    'cell_to_token_map_max': int(cell_to_token_map.max().item()) if cell_to_token_map.numel() > 0 else -1,
                    'anchor_idx': anchor_idx.detach().cpu().tolist(),
                    'nodes_after_merge': mean_stats.get('nodes_after_merge', 0.0),
                    'created_nodes': mean_stats.get('created_nodes', 0.0),
                    'updated_nodes': mean_stats.get('updated_nodes', 0.0),
                }
        else:
            self.topo_debug_cache = None

        return direction, progress, pred_goals, target_logits, emb_frames + emb_directions, mean_stats
