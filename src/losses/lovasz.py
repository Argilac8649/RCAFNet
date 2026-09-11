"""Lovasz-Softmax loss for semantic segmentation.

This implementation follows the standard Lovasz-Softmax formulation used as a
surrogate objective for optimizing mean IoU.  It accepts logits of shape
``[B, C, H, W]`` and target labels of shape ``[B, H, W]``.
"""

from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


ClassMode = Literal["all", "present"]


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t. sorted errors.

    Args:
        gt_sorted: Ground-truth foreground indicators sorted by prediction
            error in descending order, shape ``[P]``.
    """
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-12)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int | None = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten probabilities and labels while removing ignored pixels."""
    if probs.ndim != 4:
        raise ValueError(f"probs should be [B, C, H, W], got {tuple(probs.shape)}.")
    if labels.ndim != 3:
        raise ValueError(f"labels should be [B, H, W], got {tuple(labels.shape)}.")
    if probs.shape[0] != labels.shape[0] or probs.shape[-2:] != labels.shape[-2:]:
        raise ValueError(
            "probs/labels shape mismatch: "
            f"probs={tuple(probs.shape)}, labels={tuple(labels.shape)}."
        )

    # [B, C, H, W] -> [B*H*W, C]
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, probs.shape[1])
    labels = labels.contiguous().view(-1)

    if ignore_index is None:
        return probs, labels

    valid = labels != int(ignore_index)
    return probs[valid], labels[valid]


def lovasz_softmax_flat(
    probs: torch.Tensor,
    labels: torch.Tensor,
    classes: ClassMode | Iterable[int] = "present",
) -> torch.Tensor:
    """Lovasz-Softmax loss on flattened probabilities.

    Args:
        probs: Class probabilities after softmax, shape ``[P, C]``.
        labels: Ground-truth labels, shape ``[P]``.
        classes: ``"present"`` averages only over classes present in labels;
            ``"all"`` averages over all classes; an iterable can select custom
            class ids.
    """
    if probs.numel() == 0:
        return probs.sum() * 0.0

    num_classes = probs.shape[1]
    if classes in ("all", "present"):
        class_to_sum = range(num_classes)
    else:
        class_to_sum = classes

    losses = []
    for class_id in class_to_sum:
        class_id = int(class_id)
        fg = (labels == class_id).to(dtype=probs.dtype)
        if classes == "present" and fg.sum() == 0:
            continue

        class_pred = probs[:, class_id]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int | None = 255,
    classes: ClassMode | Iterable[int] = "present",
    per_image: bool = False,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss.

    Args:
        logits: Raw segmentation logits, shape ``[B, C, H, W]``.
        labels: Ground-truth labels, shape ``[B, H, W]``.
        ignore_index: Label id ignored by the loss.
        classes: Which classes to average over. ``"present"`` is common for
            semantic segmentation because absent classes do not contribute.
        per_image: If true, compute Lovasz per image and then average.
    """
    probs = F.softmax(logits, dim=1)

    if per_image:
        losses = []
        for prob, label in zip(probs, labels):
            prob_flat, label_flat = flatten_probs(prob.unsqueeze(0), label.unsqueeze(0), ignore_index)
            losses.append(lovasz_softmax_flat(prob_flat, label_flat, classes=classes))
        return torch.stack(losses).mean() if losses else probs.sum() * 0.0

    probs_flat, labels_flat = flatten_probs(probs, labels, ignore_index)
    return lovasz_softmax_flat(probs_flat, labels_flat, classes=classes)


class CrossEntropyLovaszLoss(nn.Module):
    """Composite loss: ``ce_weight * CE + lovasz_weight * LovaszSoftmax``."""

    def __init__(
        self,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        ce_weight: float = 1.0,
        lovasz_weight: float = 0.5,
        lovasz_classes: ClassMode | Iterable[int] = "present",
        lovasz_per_image: bool = False,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.lovasz_weight = float(lovasz_weight)
        self.lovasz_classes = lovasz_classes
        self.lovasz_per_image = bool(lovasz_per_image)
        self.ce = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=self.ignore_index,
            label_smoothing=float(label_smoothing),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = logits.sum() * 0.0
        if self.ce_weight != 0.0:
            loss = loss + self.ce_weight * self.ce(logits, targets)
        if self.lovasz_weight != 0.0:
            loss = loss + self.lovasz_weight * lovasz_softmax(
                logits,
                targets,
                ignore_index=self.ignore_index,
                classes=self.lovasz_classes,
                per_image=self.lovasz_per_image,
            )
        return loss
