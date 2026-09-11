"""CBAM-like dual-modal feature recalibration modules.

This file collects a lightweight channel-spatial recalibration block for
ablation/validation configs. It predicts joint dual-modal attention masks, then
applies modality-specific residual updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualModalFeatureRecalibration(nn.Module):
    """CBAM-like dual-modal channel-spatial feature recalibration.

    The module first predicts channel attention from pooled Event/Image
    features, then predicts separate spatial masks for the two modality streams.
    The learnable ``gamma`` parameters are per-channel and zero-initialized for
    conservative residual-start behavior.
    """

    def __init__(self, dim_event, dim_rgb, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.dim_event = int(dim_event)
        self.dim_rgb = int(dim_rgb)
        if self.dim_event <= 0 or self.dim_rgb <= 0:
            raise ValueError(
                f"dim_event/dim_rgb should be positive, got "
                f"{dim_event}/{dim_rgb}."
            )

        self.reduction = max(int(reduction), 1)
        spatial_kernel_size = int(spatial_kernel_size)
        if spatial_kernel_size <= 0 or spatial_kernel_size % 2 == 0:
            raise ValueError(
                "spatial_kernel_size should be a positive odd integer, "
                f"got {spatial_kernel_size}."
            )

        total_dim = self.dim_event + self.dim_rgb
        hidden_dim = max(total_dim // self.reduction, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(total_dim, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, total_dim, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            4,
            2,
            spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False,
        )
        self.gamma_event = nn.Parameter(
            torch.zeros(self.dim_event, 1, 1),
            requires_grad=True,
        )
        self.gamma_rgb = nn.Parameter(
            torch.zeros(self.dim_rgb, 1, 1),
            requires_grad=True,
        )

    def forward(self, event, rgb):
        if event.ndim != 4 or rgb.ndim != 4:
            raise ValueError(
                f"DualModalFeatureRecalibration expects NCHW tensors, got "
                f"event={tuple(event.shape)}, rgb={tuple(rgb.shape)}."
            )
        if event.shape[0] != rgb.shape[0] or event.shape[2:] != rgb.shape[2:]:
            raise ValueError(
                "DualModalFeatureRecalibration expects matched batch/spatial shapes, "
                f"got event={tuple(event.shape)}, rgb={tuple(rgb.shape)}."
            )
        if event.shape[1] != self.dim_event or rgb.shape[1] != self.dim_rgb:
            raise ValueError(
                "DualModalFeatureRecalibration input channels do not match configured dims, "
                f"got event={event.shape[1]}, rgb={rgb.shape[1]}, "
                f"expected event={self.dim_event}, rgb={self.dim_rgb}."
            )

        dual_modal = torch.cat([event, rgb], dim=1)
        channel_attn = self.sigmoid(
            self.channel_mlp(self.avg_pool(dual_modal))
            + self.channel_mlp(self.max_pool(dual_modal))
        )
        event_recalibrated = event * channel_attn[:, :self.dim_event, :, :]
        rgb_recalibrated = rgb * channel_attn[:, self.dim_event:, :, :]

        spatial_descriptor = torch.cat(
            [
                torch.mean(event_recalibrated, dim=1, keepdim=True),
                torch.max(event_recalibrated, dim=1, keepdim=True)[0],
                torch.mean(rgb_recalibrated, dim=1, keepdim=True),
                torch.max(rgb_recalibrated, dim=1, keepdim=True)[0],
            ],
            dim=1,
        )
        spatial_attn = self.sigmoid(self.spatial_conv(spatial_descriptor))
        event_out = (
            event_recalibrated
            * spatial_attn[:, 0:1, :, :]
            * self.gamma_event
            + event
        )
        rgb_out = (
            rgb_recalibrated
            * spatial_attn[:, 1:2, :, :]
            * self.gamma_rgb
            + rgb
        )
        return event_out, rgb_out


FeatureEnhance = DualModalFeatureRecalibration


__all__ = [
    "DualModalFeatureRecalibration",
    "FeatureEnhance",
]
