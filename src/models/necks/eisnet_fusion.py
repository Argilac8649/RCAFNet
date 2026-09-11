"""EISNet fusion modules.

This file keeps the original EISNet module structure separate from CAREFNet's
QAFRM/CAFFM implementation:
  - AEIM: Activity-Aware Event Integration Module
  - MRFM: Modality Recalibration and Fusion Module
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class AEIM(nn.Module):
    """Activity-Aware Event Integration Module from EISNet.

    EISNet AET is split as:
      - ``ev``: signed temporal voxel grid, [B, D, H, W]
      - ``activity``: activity map, [B, D, H, W]

    AEIM extracts multi-scale activity cues from each activity channel and uses
    them to recalibrate the corresponding event voxel channel.
    """

    def __init__(self, in_dim=1, out_dim=32):
        super().__init__()
        self.in_dim = int(in_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_dim, out_dim, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=4, padding=2),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.attn = SpatialAttention(kernel_size=7)

    def forward(self, ev, activity):
        if ev.ndim != 4 or activity.ndim != 4:
            raise ValueError(
                f"AEIM expects NCHW tensors, got ev={tuple(ev.shape)}, "
                f"activity={tuple(activity.shape)}."
            )
        if ev.shape != activity.shape:
            raise ValueError(
                "AEIM expects signed voxel and activity maps with the same "
                f"shape, got ev={tuple(ev.shape)}, activity={tuple(activity.shape)}."
            )

        batch, channels, height, width = ev.shape
        activity = activity.reshape(batch * channels, self.in_dim, height, width)

        map1 = self.stem(activity)
        _, _, h1, w1 = map1.shape
        map2 = self.pool1(map1)
        map2 = F.interpolate(map2, size=(h1, w1), mode="bilinear", align_corners=False)
        map3 = self.pool2(map1)
        map3 = F.interpolate(map3, size=(h1, w1), mode="bilinear", align_corners=False)

        activity_feature = self.fusion(map1 + map2 + map3)
        mask = self.attn(activity_feature)
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
        mask = mask.reshape(batch, channels, height, width)
        return ev * mask + ev


class EfficientCrossAttention(nn.Module):
    """Efficient cross-attention used by EISNet MRFM."""

    def __init__(self, in_channels_x, in_channels_y, key_channels, head_count, value_channels):
        super().__init__()
        if key_channels % head_count != 0 or value_channels % head_count != 0:
            raise ValueError(
                "key_channels/value_channels must be divisible by head_count, "
                f"got key={key_channels}, value={value_channels}, heads={head_count}."
            )
        self.key_channels = int(key_channels)
        self.head_count = int(head_count)
        self.value_channels = int(value_channels)
        self.keys = nn.Conv2d(in_channels_y, key_channels, 1)
        self.queries = nn.Conv2d(in_channels_x, key_channels, 1)
        self.values = nn.Conv2d(in_channels_y, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels_x, 1)

    def forward(self, x, y):
        batch, _, height, width = x.size()
        keys = self.keys(y).reshape(batch, self.key_channels, height * width)
        queries = self.queries(x).reshape(batch, self.key_channels, height * width)
        values = self.values(y).reshape(batch, self.value_channels, height * width)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(
                keys[:, i * head_key_channels:(i + 1) * head_key_channels, :],
                dim=2,
            )
            query = F.softmax(
                queries[:, i * head_key_channels:(i + 1) * head_key_channels, :],
                dim=1,
            )
            value = values[:, i * head_value_channels:(i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(
                batch,
                head_value_channels,
                height,
                width,
            )
            attended_values.append(attended_value)

        return self.reprojection(torch.cat(attended_values, dim=1))


class LayerNorm(nn.Module):
    """LayerNorm supporting NCHW tensors."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data_format={data_format!r}.")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
        )
        self.act = act_layer or nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MRFM(nn.Module):
    """Modality Recalibration and Fusion Module from EISNet."""

    def __init__(self, dim=(32, 64), head_count=4, reduction=16):
        super().__init__()
        self.dim_ev, self.dim_img = [int(v) for v in dim]
        self.reduction = max(int(reduction), 1)

        self.sigmoid = nn.Sigmoid()
        total_dim = self.dim_ev + self.dim_img
        hidden_dim = max(total_dim // self.reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(total_dim, hidden_dim, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, total_dim, 1, bias=False),
        )
        self.conv = nn.Conv2d(4, 2, 7, padding=7 // 2, bias=False)
        self.gamma_ev = nn.Parameter(torch.zeros(self.dim_ev, 1, 1), requires_grad=True)
        self.gamma_img = nn.Parameter(torch.zeros(self.dim_img, 1, 1), requires_grad=True)

        self.proj = (
            nn.Conv2d(self.dim_ev, self.dim_img, kernel_size=1, bias=False)
            if self.dim_ev != self.dim_img
            else nn.Identity()
        )
        self.norm_ev = LayerNorm(normalized_shape=self.dim_img, data_format="channels_first")
        self.norm_img = LayerNorm(normalized_shape=self.dim_img, data_format="channels_first")
        self.i2e = EfficientCrossAttention(
            in_channels_x=self.dim_img,
            in_channels_y=self.dim_img,
            key_channels=self.dim_img,
            head_count=head_count,
            value_channels=self.dim_img,
        )
        self.e2i = EfficientCrossAttention(
            in_channels_x=self.dim_img,
            in_channels_y=self.dim_img,
            key_channels=self.dim_img,
            head_count=head_count,
            value_channels=self.dim_img,
        )

        gate_hidden = max(self.dim_img * 2 // self.reduction, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(self.dim_img * 2, gate_hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1),
        )
        self.norm_ffn = LayerNorm(normalized_shape=self.dim_img, data_format="channels_first")
        self.ffn = FFN(in_features=self.dim_img, hidden_features=max(self.dim_img // 4, 1))

    def forward(self, ev, img):
        if ev.shape[0] != img.shape[0] or ev.shape[2:] != img.shape[2:]:
            raise ValueError(
                "MRFM expects matched batch/spatial shapes, "
                f"got ev={tuple(ev.shape)}, img={tuple(img.shape)}."
            )
        if ev.shape[1] != self.dim_ev or img.shape[1] != self.dim_img:
            raise ValueError(
                "MRFM input channels do not match configured dims, "
                f"got ev={ev.shape[1]}, img={img.shape[1]}, "
                f"expected ev={self.dim_ev}, img={self.dim_img}."
            )

        _, event_channels, _, _ = ev.shape

        mm = torch.cat([ev, img], dim=1)
        ca = self.sigmoid(self.fc(self.avg_pool(mm)) + self.fc(self.max_pool(mm)))
        ev_rec = ev * ca[:, :event_channels, :, :]
        img_rec = img * ca[:, event_channels:, :, :]

        sa = torch.cat(
            [
                torch.mean(ev_rec, dim=1, keepdim=True),
                torch.max(ev_rec, dim=1, keepdim=True)[0],
                torch.mean(img_rec, dim=1, keepdim=True),
                torch.max(img_rec, dim=1, keepdim=True)[0],
            ],
            dim=1,
        )
        sa = self.sigmoid(self.conv(sa))
        ev_rec = ev_rec * sa[:, 0:1, :, :] * self.gamma_ev + ev
        img_rec = img_rec * sa[:, 1:2, :, :] * self.gamma_img + img
        ev_out, img_out = ev_rec, img_rec

        ev_rec_p = self.proj(ev_rec)
        ev_f = self.norm_ev(ev_rec_p)
        img_f = self.norm_img(img_rec)
        ev_f = self.i2e(ev_f, img_f) + ev_rec_p
        img_f = self.e2i(img_f, ev_f) + img_rec

        gate = self.gate(torch.cat([ev_f, img_f], dim=1))
        out = ev_rec_p * gate[:, 0:1, :, :] + img_rec * gate[:, 1:2, :, :]
        out = self.ffn(self.norm_ffn(out)) + out
        return out, ev_out, img_out
