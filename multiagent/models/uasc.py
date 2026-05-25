"""Lightweight uncertainty-aware stage and stop controller."""

import torch
from torch import nn
from torch.nn import functional as F


IGNORE_LABEL = -100.0


def _first_tensor(*values):
    for value in values:
        if torch.is_tensor(value):
            return value
        if isinstance(value, dict):
            tensor = _first_tensor(*value.values())
            if tensor is not None:
                return tensor
    return None


def _batch_size_from(*values):
    tensor = _first_tensor(*values)
    if tensor is None or tensor.dim() == 0:
        return 1
    return int(tensor.shape[0])


def _as_column(value, batch_size, reference, fill_value=0.0):
    if value is None:
        return reference.new_full((batch_size, 1), fill_value)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    else:
        value = value.to(device=reference.device, dtype=reference.dtype)
    if value.dim() == 0:
        return value.reshape(1, 1).expand(batch_size, 1)
    value = value.reshape(value.shape[0], -1)
    if value.shape[0] == 1 and batch_size != 1:
        value = value.expand(batch_size, -1)
    if value.shape[0] != batch_size:
        return reference.new_full((batch_size, 1), fill_value)
    return value[:, :1]


def _as_xy(value, batch_size, reference):
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    else:
        value = value.to(device=reference.device, dtype=reference.dtype)
    if value.dim() == 1:
        value = value.reshape(1, -1)
    value = value.reshape(value.shape[0], -1)
    if value.shape[0] == 1 and batch_size != 1:
        value = value.expand(batch_size, -1)
    if value.shape[0] != batch_size or value.shape[1] < 2:
        return None
    return value[:, :2]


def _distance(lhs, rhs, batch_size, reference):
    lhs = _as_xy(lhs, batch_size, reference)
    rhs = _as_xy(rhs, batch_size, reference)
    if lhs is None or rhs is None:
        return reference.new_zeros((batch_size, 1))
    return torch.norm(lhs - rhs, dim=-1, keepdim=True)


def masked_bce_loss(pred, target, ignore_label=IGNORE_LABEL):
    """Binary cross entropy that ignores pseudo-label slots marked as -100."""

    if target is None:
        return pred.sum() * 0.0
    target = _as_column(target, pred.shape[0], pred)
    valid = target != float(ignore_label)
    if not torch.any(valid):
        return pred.sum() * 0.0
    return F.binary_cross_entropy(pred[valid], target[valid])


def build_uasc_labels(
    pred_target,
    current_pos,
    goal_pos,
    progress_gt=None,
    step_idx=None,
    final_step=None,
    success_radius=20.0,
    stage_radius=30.0,
):
    """Build safe UASC pseudo labels from rollout tensors when available."""

    source = _first_tensor(pred_target, current_pos, goal_pos, progress_gt, step_idx, final_step)
    if source is None:
        source = torch.zeros(1, 1)
    if not source.is_floating_point():
        source = source.to(torch.float32)
    batch_size = _batch_size_from(pred_target, current_pos, goal_pos, progress_gt, step_idx, final_step)
    reference = source.new_zeros((batch_size, 1))
    ignored = reference.new_full((batch_size, 1), IGNORE_LABEL)

    goal_xy = _as_xy(goal_pos, batch_size, reference)
    if goal_xy is None:
        return {
            "conf_label": ignored.clone(),
            "stage_label": ignored.clone(),
            "stop_label": ignored.clone(),
        }

    pred_xy = _as_xy(pred_target, batch_size, reference)
    current_xy = _as_xy(current_pos, batch_size, reference)

    if pred_xy is None:
        conf_label = ignored.clone()
    else:
        target_error = torch.norm(pred_xy - goal_xy, dim=-1, keepdim=True)
        conf_label = (target_error <= float(success_radius)).to(reference.dtype)

    if current_xy is None:
        stage_label = ignored.clone()
    else:
        goal_distance = torch.norm(current_xy - goal_xy, dim=-1, keepdim=True)
        stage_label = (goal_distance <= float(stage_radius)).to(reference.dtype)

    if progress_gt is not None:
        stop_label = (_as_column(progress_gt, batch_size, reference) >= 0.95).to(reference.dtype)
    elif step_idx is not None and final_step is not None:
        step_col = _as_column(step_idx, batch_size, reference)
        final_col = _as_column(final_step, batch_size, reference)
        stop_label = (step_col == final_col).to(reference.dtype)
    elif current_xy is not None:
        goal_distance = torch.norm(current_xy - goal_xy, dim=-1, keepdim=True)
        stop_label = (goal_distance <= float(success_radius)).to(reference.dtype)
    else:
        stop_label = ignored.clone()

    return {
        "conf_label": conf_label,
        "stage_label": stage_label,
        "stop_label": stop_label,
    }


class UASCController(nn.Module):
    """Auxiliary confidence, stage transition, and stop heads for HETT."""

    def __init__(
        self,
        hidden_size,
        dropout=0.1,
        aux_dim=9,
        lambda_conf=0.2,
        lambda_stage=0.2,
        lambda_stop=0.5,
        lambda_calib=0.1,
        use_calib=False,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.aux_dim = int(aux_dim)
        self.lambda_conf = float(lambda_conf)
        self.lambda_stage = float(lambda_stage)
        self.lambda_stop = float(lambda_stop)
        self.lambda_calib = float(lambda_calib)
        self.use_calib = bool(use_calib)

        self.aux_proj = nn.Linear(self.aux_dim, self.hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.conf_head = nn.Linear(self.hidden_size, 1)
        self.stage_head = nn.Linear(self.hidden_size, 1)
        self.stop_head = nn.Linear(self.hidden_size, 1)

    def _reference(self, *values):
        tensor = _first_tensor(*values)
        if tensor is None:
            tensor = self.aux_proj.weight.new_zeros((1, 1))
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        return tensor

    def _pool_feature(self, feat, batch_size, reference):
        if feat is None:
            return reference.new_zeros((batch_size, self.hidden_size))
        feat = feat.to(device=reference.device, dtype=reference.dtype)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.dim() > 2:
            feat = feat.reshape(feat.shape[0], -1, feat.shape[-1]).mean(dim=1)
        if feat.shape[0] == 1 and batch_size != 1:
            feat = feat.expand(batch_size, -1)
        if feat.shape[0] != batch_size or feat.shape[-1] != self.hidden_size:
            return reference.new_zeros((batch_size, self.hidden_size))
        return feat

    def _build_aux_feats(
        self,
        batch_size,
        reference,
        progress,
        pred_target,
        current_pos,
        goal_pos,
        align_conf,
        align_entropy,
        topo_conf,
        topo_coverage,
        topo_goal_rel,
    ):
        scalar_feats = [
            _as_column(progress, batch_size, reference),
            _distance(current_pos, pred_target, batch_size, reference),
            _distance(current_pos, goal_pos, batch_size, reference),
            _distance(pred_target, goal_pos, batch_size, reference),
            _as_column(align_conf, batch_size, reference),
            _as_column(align_entropy, batch_size, reference),
            _as_column(topo_conf, batch_size, reference),
            _as_column(topo_coverage, batch_size, reference),
            _as_column(topo_goal_rel, batch_size, reference),
        ]
        aux_feats = torch.cat(scalar_feats, dim=-1)
        if aux_feats.shape[-1] < self.aux_dim:
            aux_feats = torch.cat(
                (aux_feats, reference.new_zeros((batch_size, self.aux_dim - aux_feats.shape[-1]))),
                dim=-1,
            )
        elif aux_feats.shape[-1] > self.aux_dim:
            aux_feats = aux_feats[:, :self.aux_dim]
        return aux_feats

    def _calibration_loss(self, coarse_conf, conf_label):
        if conf_label is None:
            return coarse_conf.sum() * 0.0
        target = _as_column(conf_label, coarse_conf.shape[0], coarse_conf)
        valid = target != IGNORE_LABEL
        if not torch.any(valid):
            return coarse_conf.sum() * 0.0
        return torch.mean((coarse_conf[valid] - target[valid]) ** 2)

    def forward(
        self,
        coarse_feat=None,
        fine_feat=None,
        progress=None,
        pred_target=None,
        current_pos=None,
        goal_pos=None,
        align_conf=None,
        align_entropy=None,
        topo_conf=None,
        topo_coverage=None,
        topo_goal_rel=None,
        labels=None,
        return_debug=False,
    ):
        reference = self._reference(
            coarse_feat,
            fine_feat,
            progress,
            pred_target,
            current_pos,
            goal_pos,
            align_conf,
            align_entropy,
            topo_conf,
            topo_coverage,
            topo_goal_rel,
            labels,
        )
        batch_size = _batch_size_from(
            coarse_feat,
            fine_feat,
            progress,
            pred_target,
            current_pos,
            goal_pos,
            align_conf,
            align_entropy,
            topo_conf,
            topo_coverage,
            topo_goal_rel,
            labels,
        )

        coarse_pooled = self._pool_feature(coarse_feat, batch_size, reference)
        fine_pooled = self._pool_feature(fine_feat, batch_size, reference)
        aux_feats = self._build_aux_feats(
            batch_size,
            reference,
            progress,
            pred_target,
            current_pos,
            goal_pos,
            align_conf,
            align_entropy,
            topo_conf,
            topo_coverage,
            topo_goal_rel,
        )
        fused = torch.cat((coarse_pooled, fine_pooled, self.aux_proj(aux_feats)), dim=-1)
        hidden = self.fusion(fused)

        coarse_conf = torch.sigmoid(self.conf_head(hidden))
        stage_prob = torch.sigmoid(self.stage_head(hidden))
        stop_prob = torch.sigmoid(self.stop_head(hidden))
        losses = {}

        if labels:
            conf_loss = masked_bce_loss(coarse_conf, labels.get("conf_label"))
            stage_loss = masked_bce_loss(stage_prob, labels.get("stage_label"))
            stop_loss = masked_bce_loss(stop_prob, labels.get("stop_label"))
            total_loss = (
                self.lambda_conf * conf_loss
                + self.lambda_stage * stage_loss
                + self.lambda_stop * stop_loss
            )
            losses.update(
                {
                    "uasc_conf": conf_loss,
                    "uasc_stage": stage_loss,
                    "uasc_stop": stop_loss,
                }
            )
            if self.use_calib:
                calib_loss = self._calibration_loss(coarse_conf, labels.get("conf_label"))
                losses["uasc_calib"] = calib_loss
                total_loss = total_loss + self.lambda_calib * calib_loss
            losses["uasc_total"] = total_loss

        debug = {}
        if return_debug:
            debug = {
                "aux_feats": aux_feats.detach(),
                "coarse_pooled": coarse_pooled.detach(),
                "fine_pooled": fine_pooled.detach(),
            }
        return {
            "coarse_conf": coarse_conf,
            "stage_prob": stage_prob,
            "stop_prob": stop_prob,
            "losses": losses,
            "debug": debug,
        }
