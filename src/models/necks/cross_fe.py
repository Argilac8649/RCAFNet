from __future__ import annotations

import torch
import torch.nn as nn


class DualModalCrossFeatureRecalibration(nn.Module):
    """Bidirectional cross-modal feature recalibration.

    Event and RGB features first predict joint channel/spatial attention. Each
    recalibrated source feature is then projected into the target channel space
    before being injected as a zero-initialized residual update.
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
        self.event_to_rgb_proj = nn.Conv2d(
            self.dim_event,
            self.dim_rgb,
            kernel_size=1,
            bias=False,
        )
        self.rgb_to_event_proj = nn.Conv2d(
            self.dim_rgb,
            self.dim_event,
            kernel_size=1,
            bias=False,
        )
        self.gamma_event_to_rgb = nn.Parameter(
            torch.zeros(1, self.dim_rgb, 1, 1),
            requires_grad=True,
        )
        self.gamma_rgb_to_event = nn.Parameter(
            torch.zeros(1, self.dim_event, 1, 1),
            requires_grad=True,
        )

    def forward(self, event, rgb):
        if event.ndim != 4 or rgb.ndim != 4:
            raise ValueError(
                f"DualModalCrossFeatureRecalibration expects NCHW tensors, got "
                f"event={tuple(event.shape)}, rgb={tuple(rgb.shape)}."
            )
        if event.shape[0] != rgb.shape[0] or event.shape[2:] != rgb.shape[2:]:
            raise ValueError(
                "DualModalCrossFeatureRecalibration expects matched batch/spatial shapes, "
                f"got event={tuple(event.shape)}, rgb={tuple(rgb.shape)}."
            )
        if event.shape[1] != self.dim_event or rgb.shape[1] != self.dim_rgb:
            raise ValueError(
                "DualModalCrossFeatureRecalibration input channels do not match configured dims, "
                f"got event={event.shape[1]}, rgb={rgb.shape[1]}, "
                f"expected event={self.dim_event}, rgb={self.dim_rgb}."
            )

        dual_modal = torch.cat([event, rgb], dim=1)
        channel_attn = self.sigmoid(
            self.channel_mlp(self.avg_pool(dual_modal))
            + self.channel_mlp(self.max_pool(dual_modal))
        )
        event_channel_attn = channel_attn[:, :self.dim_event, :, :]
        rgb_channel_attn = channel_attn[:, self.dim_event:, :, :]
        event_recalibrated = event * event_channel_attn
        rgb_recalibrated = rgb * rgb_channel_attn

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
        event_to_rgb_spatial_attn = spatial_attn[:, 0:1, :, :]
        rgb_to_event_spatial_attn = spatial_attn[:, 1:2, :, :]

        event_to_rgb_update = (
            self.event_to_rgb_proj(event_recalibrated)
            * event_to_rgb_spatial_attn
        )
        rgb_to_event_update = (
            self.rgb_to_event_proj(rgb_recalibrated)
            * rgb_to_event_spatial_attn
        )

        rgb_out = rgb + self.gamma_event_to_rgb * event_to_rgb_update
        event_out = event + self.gamma_rgb_to_event * rgb_to_event_update

        return event_out, rgb_out


CrossFeatureRecalibration = DualModalCrossFeatureRecalibration


__all__ = [
    "DualModalCrossFeatureRecalibration",
    "CrossFeatureRecalibration",
]
