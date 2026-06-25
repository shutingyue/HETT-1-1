import math

import torch
from .enc_visual import FeatureFlat
from .enc_vl import EncoderVL
from .encodings import DatasetLearnedEncoding
from . import model_util
from torch import nn
from torch.nn import functional as F

import numpy as np

from .goal_predictor import MapEncoder


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
            stop_contrast_visual_dim = self.args.demb
            if stop_contrast_source in ("global_attn", "fixed_partition"):
                stop_contrast_visual_dim = getattr(self.args, "stop_visual_context_dim", self.args.demb)
            self.stop_contrast_loss = StopContrastLoss(
                visual_source=stop_contrast_source,
                hidden_dim=self.args.demb,
                visual_dim=stop_contrast_visual_dim,
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

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        output = {}
        emb_lang = inputs["lang"]

        map_feat = self.map_encoder(inputs['maps'])

        emb_candidates = self.candidate_encoder(inputs['candidates']) * emb_lang[:, :1, :]
        # print(torch.isnan(map_feat).any(), torch.isinf(map_feat).any())

        # # embed frames and direiction (650,49) --> 768
        # im_feature = inputs["frames"]
        # embed_frame, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:,-1, :, :])
        # h_sali = self.fc(embed_frame).view(-1,1,8,8)
        # pred_saliency = nn.functional.interpolate(h_sali,size=(224,224),mode='bilinear',align_corners=False)
        # frames_pad_emb = self.vis_feat(im_feature.view(-1, 650,7,7)).view(*im_feature.shape[:2], -1)

        # embed frames and direiction (1,49) --> 768
        im_feature = inputs["frames"]
        att_frame_feature = torch.zeros((im_feature.shape[0], 0, 49)).cuda()
        for i in range(im_feature.shape[1]):
            att_single_frame_feature, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:, i, :, :])
            att_frame_feature = torch.concat((att_frame_feature, att_single_frame_feature.unsqueeze(1)), axis=1)

        original_emb_frames = self.fc2(att_frame_feature.view(-1, 49)).view(*im_feature.shape[:2], -1)
        emb_frames = original_emb_frames
        model_stats = {}
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
                    model_stats[key] = float(value)
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

            region_context, _ = self.region_prompt_adapter.select_region_context(
                region_tokens_all,
                inputs["lang_cls"],
            )
            region_context_scaled = self._scale_region_context(region_context, original_emb_frames)
            if self.args.region_prompt_mode == "replace":
                emb_frames = region_context_scaled
            elif self.args.region_prompt_mode == "residual":
                emb_frames = original_emb_frames + self.args.region_prompt_alpha * region_context_scaled
            if use_region_attn_diversity:
                if region_attn_diversity_losses:
                    region_attn_diversity_loss = torch.stack(
                        [loss.reshape(()) for loss in region_attn_diversity_losses]
                    ).mean()
                else:
                    region_attn_diversity_loss = im_feature.sum() * 0.0
                model_stats["region_attn_diversity_loss"] = region_attn_diversity_loss

        emb_maps = self.fc_map(map_feat).view(im_feature.shape[0], -1, 768)
        # print('sss', emb_frames.shape, emb_maps.shape)
        # print(map_feat.shape)

        emb_directions = self.direction_embedding(inputs["directions"].view(-1, 4)).view(im_feature.shape[0], -1,
                                                                                         768)  # (batch, embedding_size)
        if self.training and use_stop_contrast and self.stop_contrast_loss is not None:
            stop_contrast_visual_context = None
            if stop_contrast_source in ("global_attn", "fixed_partition"):
                stop_contrast_visual_context = stop_visual_context
            elif stop_contrast_source == "region_prompt":
                stop_contrast_visual_context = region_context
            action_hidden = emb_frames + emb_directions
            model_stats["stop_contrast_score"] = self.stop_contrast_loss.score(
                action_hidden=action_hidden,
                instruction=emb_lang[:, 0, :],
                visual_context=stop_contrast_visual_context,
                detach_visual=bool(getattr(self.args, "stop_contrast_detach_visual", False)),
            )
        batch_size = emb_lang.shape[0]

        grid_map_input = torch.zeros(batch_size, self.args.grid_size ** 2, 768).cuda()

        text_fts = self.text_proj(emb_lang).permute(0, 2, 1)
        grid_masks = [[] for b in range(batch_size)]
        max_cell_num = self.args.grid_size ** 2
        grid_fts = inputs['grid_fts']
        grid_map_indexs = inputs['grid_index']
        for b in range(batch_size):
            tmp_fts = grid_fts[b].to(torch.float32)
            grid_fts_weight, _ = (tmp_fts @ text_fts[b]).max(dim=-1)
            tmp_fts = self.grid_proj(tmp_fts)

            for i in range(self.args.grid_size ** 2):
                cell_fts = tmp_fts[grid_map_indexs[b] == i]
                if cell_fts.shape[0] == 0:
                    grid_masks[b].append(0)
                else:
                    grid_masks[b].append(1)
                grid_map_input[b, i] = (
                        cell_fts * torch.softmax(grid_fts_weight[grid_map_indexs[b] == i], dim=-1).unsqueeze(
                    -1)).sum(-2)

            # if max_cell_num < sum(grid_masks[b]):
            #     max_cell_num = sum(grid_masks[b])
        # grid_masks = torch.tensor(grid_masks).cuda()
        grid_map_embeds = torch.zeros(batch_size, max_cell_num, 768).to(grid_fts[0].device)

        emb_candidates = emb_candidates + grid_map_input

        # emb_centroids = (self.centroid_encoder(inputs['centroids']) * emb_lang[:, 0, :]).view(im_feature.shape[0], -1, 768)
        # emb_centroids = self.centroid_encoder(inputs['centroids']).view(im_feature.shape[0], -1, 768)
        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl.forward_with_map(
            emb_lang,
            emb_frames,
            emb_directions,
            emb_maps,
            emb_candidates,

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

        # decoder_input = emb_directions[:,-1].reshape(-1, self.args.demb)
        output = self.decoder_2_action_full(action_decoder_input)
        # goal_logits = self.decoder_2_goal_full(goal_decoder_input)
        pred_goals = self.decoder_2_goal_full(goal_decoder_input)
        norm = torch.norm(output, dim=1, keepdim=True) + 1e-6  # 避免除以零
        direction = output / norm

        progress = self.decoder_2_progress_full(decoder_input)

        target_logits = self.decoder_2_logits_full(target_decoder_input)
        # print(encoder_out_candidates.shape)

        # print(direction, progress, goal_logits)

        if model_stats:
            return (
                direction,
                progress,
                pred_goals,
                target_logits,
                emb_frames + emb_directions,
                model_stats,
            )
        return direction, progress, pred_goals, target_logits, emb_frames + emb_directions
