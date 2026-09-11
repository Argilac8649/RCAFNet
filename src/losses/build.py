"""Loss builders."""

from __future__ import annotations

import torch
import torch.nn as nn

from .lovasz import CrossEntropyLovaszLoss


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def build_loss(config) -> nn.Module:
    """Build segmentation loss from config.

    Supported ``loss.type`` values:
        - ``ce`` / ``cross_entropy``
        - ``ce_lovasz`` / ``ce+lovasz`` / ``cross_entropy_lovasz``
    """
    loss_cfg = _cfg_get(config, "loss", None)
    loss_type = str(_cfg_get(loss_cfg, "type", "ce")).strip().lower()
    ignore_index = int(_cfg_get(config, "ignore_index", 255))
    label_smoothing = float(_cfg_get(config, "label_smoothing", 0.0))

    if loss_type in {"ce", "cross_entropy", "crossentropy"}:
        return torch.nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    if loss_type in {
        "ce_lovasz",
        "ce+lovasz",
        "ce_lovasz_softmax",
        "cross_entropy_lovasz",
        "cross_entropy_lovasz_softmax",
    }:
        return CrossEntropyLovaszLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            ce_weight=float(_cfg_get(loss_cfg, "ce_weight", 1.0)),
            lovasz_weight=float(_cfg_get(loss_cfg, "lovasz_weight", 0.5)),
            lovasz_classes=_cfg_get(loss_cfg, "lovasz_classes", "present"),
            lovasz_per_image=bool(_cfg_get(loss_cfg, "lovasz_per_image", False)),
        )

    raise ValueError(
        f"Unknown loss.type='{loss_type}'. Available: ce, ce_lovasz."
    )
