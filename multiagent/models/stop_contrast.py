import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class StopContrastLoss(nn.Module):
    """Parameterized auxiliary StopContrast scorer and BCE loss helper."""

    def __init__(
        self,
        visual_source="none",
        hidden_dim=768,
        visual_dim=768,
        instruction_dim=49,
        proj_dim=256,
        temperature=0.07,
        dropout=0.1,
        require_both_pos_neg=True,
    ):
        super().__init__()
        if visual_source not in ("none", "global_attn", "fixed_partition", "region_prompt"):
            raise ValueError(f"Unsupported stop contrast visual source: {visual_source}.")
        self.visual_source = visual_source
        self.hidden_dim = int(hidden_dim)
        self.visual_dim = int(visual_dim)
        self.instruction_dim = int(instruction_dim)
        self.proj_dim = int(proj_dim)
        self.temperature = float(temperature)
        self.require_both_pos_neg = bool(require_both_pos_neg)

        stop_input_dim = self.hidden_dim if visual_source == "none" else self.hidden_dim + self.visual_dim
        self.stop_encoder = nn.Sequential(
            nn.Linear(stop_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.proj_dim),
        )
        self.legacy_instruction_proj = nn.Linear(self.instruction_dim, self.hidden_dim)
        self.embed_instruction_proj = nn.Identity()
        self.instr_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.proj_dim),
        )

    def forward(
        self,
        action_hidden,
        instruction,
        progress_target=None,
        valid_mask=None,
        visual_context=None,
        progress_threshold=0.8,
        detach_visual=False,
        positive_mode="progress_threshold",
        strict_pos_threshold=0.95,
        hard_neg_min=0.80,
        easy_neg_max=0.50,
        use_easy_negatives=False,
        ignore_ambiguous=True,
    ):
        scores = self.score(
            action_hidden=action_hidden,
            instruction=instruction,
            visual_context=visual_context,
            detach_visual=detach_visual,
        )
        if progress_target is None:
            zero = scores.sum() * 0.0
            return zero, self.empty_diagnostics(zero), scores
        loss, diagnostics = self.compute_loss_from_scores(
            scores=scores,
            progress_target=progress_target,
            valid_mask=valid_mask,
            progress_threshold=progress_threshold,
            positive_mode=positive_mode,
            strict_pos_threshold=strict_pos_threshold,
            hard_neg_min=hard_neg_min,
            easy_neg_max=easy_neg_max,
            use_easy_negatives=use_easy_negatives,
            ignore_ambiguous=ignore_ambiguous,
        )
        return loss, diagnostics, scores

    def score(self, action_hidden, instruction, visual_context=None, detach_visual=False):
        if action_hidden.dim() != 3:
            raise ValueError(f"action_hidden must be [B, T, D], got {tuple(action_hidden.shape)}.")
        if action_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected action hidden dim {self.hidden_dim}, got {action_hidden.shape[-1]}.")

        stop_repr = action_hidden
        if self.visual_source != "none":
            if visual_context is None:
                raise ValueError(f"visual_context is required for source={self.visual_source}.")
            if visual_context.shape[:2] != action_hidden.shape[:2]:
                raise ValueError(
                    "visual_context must share [B, T] with action_hidden, "
                    f"got {tuple(visual_context.shape)} and {tuple(action_hidden.shape)}."
                )
            if visual_context.shape[-1] != self.visual_dim:
                raise ValueError(f"Expected visual context dim {self.visual_dim}, got {visual_context.shape[-1]}.")
            if detach_visual:
                visual_context = visual_context.detach()
            stop_repr = torch.cat([action_hidden, visual_context], dim=-1)

        instruction_repr = self._project_instruction(instruction)
        stop_proj = F.normalize(self.stop_encoder(stop_repr), dim=-1, eps=1e-6)
        instr_proj = F.normalize(self.instr_encoder(instruction_repr), dim=-1, eps=1e-6)
        scores = torch.sum(stop_proj * instr_proj.unsqueeze(1), dim=-1)
        return scores / max(self.temperature, 1e-6)

    def _project_instruction(self, instruction):
        if instruction is None:
            raise ValueError("instruction is required for StopContrastLoss.")
        if instruction.dim() != 2:
            raise ValueError(f"instruction must be [B, 49] or [B, 768], got {tuple(instruction.shape)}.")
        if instruction.shape[-1] == self.instruction_dim:
            return self.legacy_instruction_proj(instruction)
        if instruction.shape[-1] == self.hidden_dim:
            return self.embed_instruction_proj(instruction)
        raise ValueError(
            "instruction last dim must match instruction_dim or hidden_dim, "
            f"got {instruction.shape[-1]}."
        )

    def compute_loss_from_scores(
        self,
        scores,
        progress_target,
        valid_mask=None,
        progress_threshold=0.8,
        require_both_pos_neg=None,
        positive_mode="progress_threshold",
        strict_pos_threshold=0.95,
        hard_neg_min=0.80,
        easy_neg_max=0.50,
        use_easy_negatives=False,
        ignore_ambiguous=True,
    ):
        if scores.dim() != 2:
            raise ValueError(f"scores must be [B, T], got {tuple(scores.shape)}.")
        if progress_target.shape != scores.shape:
            raise ValueError(
                "progress_target must match scores [B, T], "
                f"got {tuple(progress_target.shape)} and {tuple(scores.shape)}."
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(scores, dtype=torch.bool)
        elif valid_mask.shape != scores.shape:
            raise ValueError(
                "valid_mask must match scores [B, T], "
                f"got {tuple(valid_mask.shape)} and {tuple(scores.shape)}."
            )
        else:
            valid_mask = valid_mask.to(dtype=torch.bool, device=scores.device)

        progress_target = progress_target.to(device=scores.device, dtype=scores.dtype)
        positive_mode = getattr(positive_mode, "lower", lambda: positive_mode)()
        if positive_mode == "conversion_aware":
            return self._compute_conversion_aware_loss_from_scores(
                scores=scores,
                progress_target=progress_target,
                valid_mask=valid_mask,
                strict_pos_threshold=strict_pos_threshold,
                hard_neg_min=hard_neg_min,
                easy_neg_max=easy_neg_max,
                use_easy_negatives=use_easy_negatives,
                ignore_ambiguous=ignore_ambiguous,
                require_both_pos_neg=require_both_pos_neg,
            )
        if positive_mode != "progress_threshold":
            raise ValueError(f"Unsupported StopContrast positive mode: {positive_mode}.")
        positive_mask = valid_mask & (progress_target >= float(progress_threshold))
        negative_mask = valid_mask & (progress_target < float(progress_threshold))
        labels = positive_mask.to(dtype=scores.dtype)

        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]
        positive_scores = scores[positive_mask]
        negative_scores = scores[negative_mask]
        num_valid = int(valid_mask.sum().item())
        num_pos = int(positive_mask.sum().item())
        num_neg = int(negative_mask.sum().item())
        global_num_valid, global_num_pos, global_num_neg = self._global_counts(
            scores,
            num_valid,
            num_pos,
            num_neg,
        )

        if require_both_pos_neg is None:
            require_both_pos_neg = self.require_both_pos_neg
        skip_reason = "none"
        if num_valid == 0:
            skip_reason = "no_valid"
        elif bool(require_both_pos_neg) and num_pos == 0:
            skip_reason = "no_pos"
        elif bool(require_both_pos_neg) and num_neg == 0:
            skip_reason = "no_neg"

        skipped = skip_reason != "none"
        any_rank_skipped = self._any_rank_skipped(scores, skipped)
        if skip_reason != "none":
            zero = scores.sum() * 0.0
            return zero, self._diagnostics(
                zero,
                positive_scores,
                negative_scores,
                num_valid,
                num_pos,
                num_neg,
                global_num_valid,
                global_num_pos,
                global_num_neg,
                skipped=skipped,
                skip_reason=skip_reason,
                any_rank_skipped=any_rank_skipped,
            )

        loss = F.binary_cross_entropy_with_logits(valid_scores, valid_labels, reduction="mean")
        return loss, self._diagnostics(
            loss,
            positive_scores,
            negative_scores,
            num_valid,
            num_pos,
            num_neg,
            global_num_valid,
            global_num_pos,
            global_num_neg,
            skipped=False,
            skip_reason=skip_reason,
            any_rank_skipped=any_rank_skipped,
        )

    def _compute_conversion_aware_loss_from_scores(
        self,
        scores,
        progress_target,
        valid_mask,
        strict_pos_threshold=0.95,
        hard_neg_min=0.80,
        easy_neg_max=0.50,
        use_easy_negatives=False,
        ignore_ambiguous=True,
        require_both_pos_neg=None,
    ):
        # Conversion-aware mode treats near-goal-but-not-stop states as hard negatives,
        # so the auxiliary loss learns stop conversion rather than only near-goal Oracle SR.
        strict_pos_threshold = float(strict_pos_threshold)
        hard_neg_min = float(hard_neg_min)
        easy_neg_max = float(easy_neg_max)
        use_easy_negatives = bool(use_easy_negatives)
        ignore_ambiguous = bool(ignore_ambiguous)

        strict_pos_mask = valid_mask & (progress_target >= strict_pos_threshold)
        hard_neg_mask = valid_mask & (progress_target >= hard_neg_min) & (progress_target < strict_pos_threshold)
        easy_neg_mask = valid_mask & (progress_target < easy_neg_max) if use_easy_negatives else torch.zeros_like(valid_mask)
        selected_neg_mask = hard_neg_mask | easy_neg_mask
        selected_mask = strict_pos_mask | selected_neg_mask
        ambiguous_mask = valid_mask & ~selected_mask
        if not ignore_ambiguous:
            selected_neg_mask = selected_neg_mask | ambiguous_mask
            selected_mask = strict_pos_mask | selected_neg_mask
        ignored_mask = valid_mask & ~selected_mask

        labels = strict_pos_mask.to(dtype=scores.dtype)
        loss_scores = scores[selected_mask]
        loss_labels = labels[selected_mask]
        positive_scores = scores[strict_pos_mask]
        negative_scores = scores[selected_neg_mask]
        hard_neg_scores = scores[hard_neg_mask]
        easy_neg_scores = scores[easy_neg_mask]

        num_valid = int(selected_mask.sum().item())
        num_pos = int(strict_pos_mask.sum().item())
        num_neg = int(selected_neg_mask.sum().item())
        num_hard_neg = int(hard_neg_mask.sum().item())
        num_easy_neg = int(easy_neg_mask.sum().item())
        num_ambiguous = int(ambiguous_mask.sum().item())
        num_ignored = int(ignored_mask.sum().item())
        global_counts = self._global_count_values(
            scores,
            [num_valid, num_pos, num_neg, num_hard_neg, num_easy_neg, num_ambiguous, num_ignored],
        )
        global_num_valid, global_num_pos, global_num_neg = global_counts[:3]

        if require_both_pos_neg is None:
            require_both_pos_neg = self.require_both_pos_neg
        skip_reason = "none"
        if num_valid == 0:
            skip_reason = "no_valid"
        elif bool(require_both_pos_neg) and num_pos == 0:
            skip_reason = "no_pos"
        elif bool(require_both_pos_neg) and num_neg == 0:
            skip_reason = "no_neg"

        skipped = skip_reason != "none"
        any_rank_skipped = self._any_rank_skipped(scores, skipped)
        if skipped:
            zero = scores.sum() * 0.0
            return zero, self._conversion_aware_diagnostics(
                zero,
                positive_scores,
                negative_scores,
                hard_neg_scores,
                easy_neg_scores,
                num_valid,
                num_pos,
                num_neg,
                num_hard_neg,
                num_easy_neg,
                num_ambiguous,
                num_ignored,
                global_counts,
                strict_pos_threshold,
                hard_neg_min,
                easy_neg_max,
                use_easy_negatives,
                skipped=skipped,
                skip_reason=skip_reason,
                any_rank_skipped=any_rank_skipped,
            )

        loss = F.binary_cross_entropy_with_logits(loss_scores, loss_labels, reduction="mean")
        return loss, self._conversion_aware_diagnostics(
            loss,
            positive_scores,
            negative_scores,
            hard_neg_scores,
            easy_neg_scores,
            num_valid,
            num_pos,
            num_neg,
            num_hard_neg,
            num_easy_neg,
            num_ambiguous,
            num_ignored,
            global_counts,
            strict_pos_threshold,
            hard_neg_min,
            easy_neg_max,
            use_easy_negatives,
            skipped=False,
            skip_reason=skip_reason,
            any_rank_skipped=any_rank_skipped,
        )

    @staticmethod
    def _global_count_values(scores, values):
        counts = torch.tensor(values, device=scores.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return [int(value.item()) for value in counts]

    @staticmethod
    def _global_counts(scores, num_valid, num_pos, num_neg):
        counts = torch.tensor(
            [num_valid, num_pos, num_neg],
            device=scores.device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return (int(counts[0].item()), int(counts[1].item()), int(counts[2].item()))

    @staticmethod
    def _any_rank_skipped(scores, skipped):
        skipped_tensor = torch.tensor(
            [1.0 if skipped else 0.0],
            device=scores.device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(skipped_tensor, op=dist.ReduceOp.MAX)
        return float(skipped_tensor.item())

    @staticmethod
    def _safe_mean_value(scores, count):
        if count <= 0:
            return 0.0
        value = scores.detach().to(torch.float32).mean()
        return float(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).item())

    @staticmethod
    def _skip_reason_id(reason):
        return {
            "none": 0.0,
            "no_valid": 1.0,
            "no_pos": 2.0,
            "no_neg": 3.0,
        }.get(reason, -1.0)

    def _visual_source_id(self):
        return {
            "none": 0.0,
            "global_attn": 1.0,
            "fixed_partition": 2.0,
            "region_prompt": 3.0,
        }.get(self.visual_source, -1.0)

    def empty_diagnostics(self, loss):
        empty_scores = loss.new_zeros(0)
        return self._diagnostics(
            loss,
            empty_scores,
            empty_scores,
            0,
            0,
            0,
            0,
            0,
            0,
            skipped=True,
            skip_reason="no_valid",
            any_rank_skipped=1.0,
        )

    def _conversion_aware_diagnostics(
        self,
        loss,
        positive_scores,
        negative_scores,
        hard_neg_scores,
        easy_neg_scores,
        num_valid,
        num_pos,
        num_neg,
        num_hard_neg,
        num_easy_neg,
        num_ambiguous,
        num_ignored,
        global_counts,
        strict_pos_threshold,
        hard_neg_min,
        easy_neg_max,
        use_easy_negatives,
        skipped=False,
        skip_reason="none",
        any_rank_skipped=0.0,
    ):
        diagnostics = self._diagnostics(
            loss,
            positive_scores,
            negative_scores,
            num_valid,
            num_pos,
            num_neg,
            global_counts[0],
            global_counts[1],
            global_counts[2],
            skipped=skipped,
            skip_reason=skip_reason,
            any_rank_skipped=any_rank_skipped,
        )
        hard_neg_mean = self._safe_mean_value(hard_neg_scores, num_hard_neg)
        easy_neg_mean = self._safe_mean_value(easy_neg_scores, num_easy_neg)
        pos_mean = diagnostics.get("stop_contrast_score_pos_mean", 0.0)
        diagnostics.update({
            "stop_contrast_positive_mode_id": 1.0,
            "stop_contrast_strict_pos_threshold": float(strict_pos_threshold),
            "stop_contrast_hard_neg_min": float(hard_neg_min),
            "stop_contrast_easy_neg_max": float(easy_neg_max),
            "stop_contrast_use_easy_negatives": 1.0 if use_easy_negatives else 0.0,
            "stop_contrast_num_strict_pos": float(num_pos),
            "stop_contrast_num_hard_neg": float(num_hard_neg),
            "stop_contrast_num_easy_neg": float(num_easy_neg),
            "stop_contrast_num_ambiguous": float(num_ambiguous),
            "stop_contrast_num_ignored": float(num_ignored),
            "stop_contrast_conversion_pos_ratio": float(num_pos) / float(max(num_valid, 1)),
            "stop_contrast_conversion_neg_ratio": float(num_neg) / float(max(num_valid, 1)),
            "stop_contrast_score_hard_neg_mean": hard_neg_mean,
            "stop_contrast_score_easy_neg_mean": easy_neg_mean,
            "stop_contrast_score_gap_pos_hard_neg": (
                pos_mean - hard_neg_mean if num_pos > 0 and num_hard_neg > 0 else 0.0
            ),
            "stop_contrast_global_num_hard_neg": float(global_counts[3]),
            "stop_contrast_global_num_easy_neg": float(global_counts[4]),
            "stop_contrast_global_num_ambiguous": float(global_counts[5]),
            "stop_contrast_global_num_ignored": float(global_counts[6]),
        })
        return diagnostics

    def _diagnostics(
        self,
        loss,
        positive_scores,
        negative_scores,
        num_valid,
        num_pos,
        num_neg,
        global_num_valid,
        global_num_pos,
        global_num_neg,
        skipped=False,
        skip_reason="none",
        any_rank_skipped=0.0,
    ):
        with torch.no_grad():
            if num_pos > 0:
                pos_mean = positive_scores.detach().to(torch.float32).mean()
                pos_mean_value = float(torch.nan_to_num(pos_mean, nan=0.0, posinf=0.0, neginf=0.0).item())
            else:
                pos_mean_value = 0.0
            if num_neg > 0:
                neg_mean = negative_scores.detach().to(torch.float32).mean()
                neg_mean_value = float(torch.nan_to_num(neg_mean, nan=0.0, posinf=0.0, neginf=0.0).item())
            else:
                neg_mean_value = 0.0
            loss_value = float(torch.nan_to_num(loss.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).item())
            pos_ratio = float(num_pos) / float(max(num_valid, 1))
            global_pos_ratio = float(global_num_pos) / float(max(global_num_valid, 1))
            global_skip_reason = "none"
            if global_num_valid == 0:
                global_skip_reason = "no_valid"
            elif global_num_pos == 0:
                global_skip_reason = "no_pos"
            elif global_num_neg == 0:
                global_skip_reason = "no_neg"
            score_gap = pos_mean_value - neg_mean_value if num_pos > 0 and num_neg > 0 else 0.0
            return {
                "stop_contrast_loss": loss_value,
                "stop_contrast_positive_mode_id": 0.0,
                "stop_contrast_num_valid": float(num_valid),
                "stop_contrast_num_pos": float(num_pos),
                "stop_contrast_num_neg": float(num_neg),
                "stop_contrast_pos_ratio": pos_ratio,
                "stop_contrast_skipped": 1.0 if skipped else 0.0,
                "stop_contrast_skip_reason_id": self._skip_reason_id(skip_reason),
                "stop_contrast_local_num_valid": float(num_valid),
                "stop_contrast_local_num_pos": float(num_pos),
                "stop_contrast_local_num_neg": float(num_neg),
                "stop_contrast_local_skipped": 1.0 if skipped else 0.0,
                "stop_contrast_local_skip_reason_id": self._skip_reason_id(skip_reason),
                "stop_contrast_any_rank_skipped": float(any_rank_skipped),
                "stop_contrast_global_num_valid": float(global_num_valid),
                "stop_contrast_global_num_pos": float(global_num_pos),
                "stop_contrast_global_num_neg": float(global_num_neg),
                "stop_contrast_global_pos_ratio": global_pos_ratio,
                "stop_contrast_global_skip_reason_id": self._skip_reason_id(global_skip_reason),
                "stop_contrast_score_pos_mean": pos_mean_value,
                "stop_contrast_score_neg_mean": neg_mean_value,
                "stop_contrast_score_gap": score_gap,
                "stop_contrast_visual_source_id": self._visual_source_id(),
                "stop_contrast_temperature": self.temperature,
            }
