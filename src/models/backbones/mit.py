"""Single-stream MixVision Transformer / SegFormer MiT backbone.

This module intentionally contains only the *single-stream* MiT feature
extractor. RGB/Event stage refinement and fusion modules live in
``src.models.necks`` and are assembled by higher-level encoders under
``src.models.encoders``.

The public backbone returns four NCHW feature maps at strides 4/8/16/32 and
also exposes ``forward_stage`` so dual-stream encoders can interleave two MiT
streams with fusion modules after each stage.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_


MaybeCheckpoint = Union[str, Path, Dict[str, torch.Tensor], None, bool]


MIT_CONFIGS: Dict[str, Dict[str, Sequence[int]]] = {
    "mit_b0": {
        "embed_dims": (32, 64, 160, 256),
        "num_heads": (1, 2, 5, 8),
        "depths": (2, 2, 2, 2),
        "sr_ratios": (8, 4, 2, 1),
    },
    "mit_b1": {
        "embed_dims": (64, 128, 320, 512),
        "num_heads": (1, 2, 5, 8),
        "depths": (2, 2, 2, 2),
        "sr_ratios": (8, 4, 2, 1),
    },
    "mit_b2": {
        "embed_dims": (64, 128, 320, 512),
        "num_heads": (1, 2, 5, 8),
        "depths": (3, 4, 6, 3),
        "sr_ratios": (8, 4, 2, 1),
    },
    "mit_b3": {
        "embed_dims": (64, 128, 320, 512),
        "num_heads": (1, 2, 5, 8),
        "depths": (3, 4, 18, 3),
        "sr_ratios": (8, 4, 2, 1),
    },
    "mit_b4": {
        "embed_dims": (64, 128, 320, 512),
        "num_heads": (1, 2, 5, 8),
        "depths": (3, 8, 27, 3),
        "sr_ratios": (8, 4, 2, 1),
    },
    "mit_b5": {
        "embed_dims": (64, 128, 320, 512),
        "num_heads": (1, 2, 5, 8),
        "depths": (3, 6, 40, 3),
        "sr_ratios": (8, 4, 2, 1),
    },
}


def _to_int_pair(value, name: str = "value") -> Tuple[int, int]:
    """Convert an int/ListConfig/list/tuple to a 2-int tuple."""
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an int or 2-element sequence, got {value!r}.")

    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an int or 2-element sequence, got {type(value)}."
        ) from exc

    if len(values) == 1:
        return int(values[0]), int(values[0])
    if len(values) != 2:
        raise ValueError(f"{name} must contain 1 or 2 values, got {values!r}.")
    return int(values[0]), int(values[1])


def _divide_pair(value, divisor: int) -> Tuple[int, int]:
    first, second = _to_int_pair(value, name="img_size")
    divisor = int(divisor)
    return max(first // divisor, 1), max(second // divisor, 1)


def _has_pretrained(pretrained: MaybeCheckpoint) -> bool:
    """Return whether a local checkpoint/state_dict is explicitly configured."""
    if pretrained is None or pretrained is False:
        return False
    if (
        isinstance(pretrained, str)
        and pretrained.strip().lower() in {"", "none", "null", "false"}
    ):
        return False
    return True


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common prefixes from plain MiT/SegFormer pretrained checkpoints."""
    prefixes = (
        "module.",
        "model.",
        "backbone.",
    )

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _load_state_dict_from_file_or_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, (str, Path)):
        raw = torch.load(str(checkpoint), map_location="cpu")
    else:
        raw = checkpoint

    if isinstance(raw, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break

    if not isinstance(raw, dict):
        raise TypeError("pretrained must be a checkpoint path or state_dict-like dict.")

    return _clean_state_dict(raw)


def adapt_input_conv_weight(weight: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    """Adapt a pretrained first convolution to a different input channel count.

    Used for grayscale RGB streams or event streams whose first patch
    embedding may expect a non-RGB channel count while ImageNet MiT weights
    are usually [C, 3, k, k].
    """
    if weight.ndim != 4 or len(target_shape) != 4:
        return None

    out_channels, in_channels, kernel_h, kernel_w = target_shape
    if (
        weight.shape[0] != out_channels
        or weight.shape[2] != kernel_h
        or weight.shape[3] != kernel_w
    ):
        return None

    if weight.shape[1] == in_channels:
        return weight

    # Follow timm's convention:
    #   * RGB -> grayscale: sum RGB kernels so a replicated grayscale image
    #     produces the same response as the original RGB conv.
    #   * RGB -> N channels: repeat kernels and scale by old_in / new_in to
    #     keep the initial activation magnitude comparable.
    source_in_channels = weight.shape[1]
    if in_channels == 1:
        return weight.sum(dim=1, keepdim=True)

    repeat = (in_channels + source_in_channels - 1) // source_in_channels
    adapted = weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
    adapted = adapted * (source_in_channels / float(in_channels))
    return adapted


def frame_type_to_event_in_chans(frame_type) -> int:
    """Return event input channels for supported event-frame layouts."""
    if isinstance(frame_type, int):
        channels = int(frame_type)
        if channels <= 0:
            raise ValueError(
                f"frame_type should define positive channels, got {channels}."
            )
        return channels

    frame_type = str(frame_type).strip().lower()
    if frame_type == "10c":
        return 10
    if frame_type == "aet":
        return 6
    raise ValueError(
        f"Unknown frame_type '{frame_type}'. Expected one of: 10c, aet."
    )


class DWConv(nn.Module):
    """Depth-wise convolution used inside MiT MLP blocks."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )

    def forward(self, x, height: int, width: int):
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width).contiguous()
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height: int, width: int):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = int(sr_ratio)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height: int, width: int):
        batch, num_tokens, channels = x.shape
        q = self.q(x).reshape(batch, num_tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            x_ = self.sr(x_).reshape(batch, channels, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
        else:
            kv = self.kv(x).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)

        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, height: int, width: int):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image/Event to overlapped patch tokens."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class MiTBackbone(nn.Module):
    """Pure single-stream MiT backbone with four feature stages."""

    def __init__(
        self,
        model_name: str = "mit_b1",
        in_chans: int = 3,
        img_size=224,
        pretrained: MaybeCheckpoint = None,
        mlp_ratios: Sequence[float] = (4.0, 4.0, 4.0, 4.0),
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()

        model_name = str(model_name).lower()
        if model_name not in MIT_CONFIGS:
            raise ValueError(
                f"Unknown MiT backbone '{model_name}'. "
                f"Available: {list(MIT_CONFIGS.keys())}."
            )
        if len(mlp_ratios) != 4:
            raise ValueError(f"mlp_ratios must have 4 values, got {mlp_ratios!r}.")

        cfg = MIT_CONFIGS[model_name]
        self.model_name = model_name
        self.in_chans = int(in_chans)
        self.embed_dims = list(cfg["embed_dims"])
        self.num_heads = list(cfg["num_heads"])
        self.depths = list(cfg["depths"])
        self.sr_ratios = list(cfg["sr_ratios"])
        self.out_channels = self.embed_dims

        img_size = _to_int_pair(img_size, name="img_size")

        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=self.in_chans,
            embed_dim=self.embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=_divide_pair(img_size, 4),
            patch_size=3,
            stride=2,
            in_chans=self.embed_dims[0],
            embed_dim=self.embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=_divide_pair(img_size, 8),
            patch_size=3,
            stride=2,
            in_chans=self.embed_dims[1],
            embed_dim=self.embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=_divide_pair(img_size, 16),
            patch_size=3,
            stride=2,
            in_chans=self.embed_dims[2],
            embed_dim=self.embed_dims[3],
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        self.block1 = self._make_stage(0, mlp_ratios[0], dpr[cur:cur + self.depths[0]], qkv_bias, qk_scale, drop_rate, attn_drop_rate, norm_layer)
        self.norm1 = norm_layer(self.embed_dims[0])
        cur += self.depths[0]
        self.block2 = self._make_stage(1, mlp_ratios[1], dpr[cur:cur + self.depths[1]], qkv_bias, qk_scale, drop_rate, attn_drop_rate, norm_layer)
        self.norm2 = norm_layer(self.embed_dims[1])
        cur += self.depths[1]
        self.block3 = self._make_stage(2, mlp_ratios[2], dpr[cur:cur + self.depths[2]], qkv_bias, qk_scale, drop_rate, attn_drop_rate, norm_layer)
        self.norm3 = norm_layer(self.embed_dims[2])
        cur += self.depths[2]
        self.block4 = self._make_stage(3, mlp_ratios[3], dpr[cur:cur + self.depths[3]], qkv_bias, qk_scale, drop_rate, attn_drop_rate, norm_layer)
        self.norm4 = norm_layer(self.embed_dims[3])

        self.apply(self._init_weights)

        if _has_pretrained(pretrained):
            if pretrained is True:
                raise ValueError(
                    "MiTBackbone does not download pretrained weights. "
                    "Pass a local checkpoint path instead."
                )
            self.init_weights(pretrained)

    def _make_stage(
        self,
        stage_idx: int,
        mlp_ratio: float,
        dpr: Sequence[float],
        qkv_bias: bool,
        qk_scale,
        drop_rate: float,
        attn_drop_rate: float,
        norm_layer,
    ) -> nn.ModuleList:
        return nn.ModuleList([
            Block(
                dim=self.embed_dims[stage_idx],
                num_heads=self.num_heads[stage_idx],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                sr_ratio=self.sr_ratios[stage_idx],
            )
            for i in range(self.depths[stage_idx])
        ])

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(
        self,
        pretrained: Union[str, Dict[str, torch.Tensor]],
        adapt_input_conv: bool = True,
    ) -> None:
        """Load matching MiT weights.

        When ``adapt_input_conv`` is true, ``patch_embed1.proj.weight`` is
        adapted from RGB pretrained weights to non-3-channel inputs.  Disable it
        to mimic the original EISNet behavior, where the first patch embedding
        is randomly initialized when the input channel count differs.
        """
        raw_state = _load_state_dict_from_file_or_dict(pretrained)
        model_state = self.state_dict()
        matching_state: Dict[str, torch.Tensor] = {}

        for key, value in raw_state.items():
            if key not in model_state:
                continue
            if model_state[key].shape == value.shape:
                matching_state[key] = value
                continue
            if adapt_input_conv and key.endswith("patch_embed1.proj.weight"):
                adapted = adapt_input_conv_weight(value, model_state[key].shape)
                if adapted is not None:
                    matching_state[key] = adapted

        self.load_state_dict(matching_state, strict=False)
        self.pretrained_loaded_keys = list(matching_state.keys())

    def _stage_modules(self, stage_idx: int):
        if stage_idx == 0:
            return self.patch_embed1, self.block1, self.norm1
        if stage_idx == 1:
            return self.patch_embed2, self.block2, self.norm2
        if stage_idx == 2:
            return self.patch_embed3, self.block3, self.norm3
        if stage_idx == 3:
            return self.patch_embed4, self.block4, self.norm4
        raise IndexError(f"stage_idx must be in [0, 3], got {stage_idx}.")

    @staticmethod
    def _run_stage(
        x: torch.Tensor,
        patch_embed: nn.Module,
        blocks: Iterable[nn.Module],
        norm: nn.Module,
    ) -> torch.Tensor:
        x, height, width = patch_embed(x)
        for block in blocks:
            x = block(x, height, width)
        x = norm(x)
        batch = x.shape[0]
        return x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward_stage(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Run one MiT stage and return the NCHW feature map."""
        patch_embed, blocks, norm = self._stage_modules(stage_idx)
        return self._run_stage(x, patch_embed, blocks, norm)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"MiTBackbone expects [B,C,H,W], got {tuple(x.shape)}.")
        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"Input channel mismatch: model expects {self.in_chans}, got {x.shape[1]}."
            )

        outs = []
        for stage_idx in range(4):
            x = self.forward_stage(x, stage_idx)
            outs.append(x)
        return outs

    def forward(self, image: torch.Tensor, event: torch.Tensor | None = None) -> List[torch.Tensor]:
        # SegNet always calls encoder(image, event); single-stream encoders use image.
        return self.forward_features(image)


class EventMiTBackbone(MiTBackbone):
    """Event-only MiT backbone for the shared ``SegNet(image, event)`` call.

    The generic ``SegNet`` wrapper always calls ``encoder(image, event)``.
    For event-only ablations this backbone ignores the RGB placeholder and
    feeds the event tensor into a single-stream MiT encoder.
    """

    def __init__(
        self,
        model_name: str = "mit_b1",
        frame_type: str = "10c",
        event_in_chans: int | None = None,
        **kwargs,
    ) -> None:
        event_in_chans = (
            frame_type_to_event_in_chans(frame_type)
            if event_in_chans is None
            else int(event_in_chans)
        )
        if event_in_chans <= 0:
            raise ValueError(
                f"event_in_chans should be positive, got {event_in_chans}."
            )
        self.frame_type = frame_type
        self.event_in_chans = event_in_chans
        super().__init__(
            model_name=model_name,
            in_chans=event_in_chans,
            **kwargs,
        )

    def forward(
        self,
        image: torch.Tensor,
        event: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        # Prefer the event tensor supplied by datasets.  Falling back to
        # ``image`` makes direct backbone debugging with a single tensor easy.
        x = event if event is not None else image
        return self.forward_features(x)


class mit_b0(MiTBackbone):
    def __init__(self, **kwargs):
        super().__init__(model_name="mit_b0", **kwargs)


class mit_b1(MiTBackbone):
    def __init__(self, **kwargs):
        super().__init__(model_name="mit_b1", **kwargs)


class mit_b2(MiTBackbone):
    def __init__(self, **kwargs):
        super().__init__(model_name="mit_b2", **kwargs)


class mit_b3(MiTBackbone):
    def __init__(self, **kwargs):
        super().__init__(model_name="mit_b3", **kwargs)


class mit_b4(MiTBackbone):
    def __init__(self, **kwargs):
        super().__init__(model_name="mit_b4", **kwargs)


class mit_b5(MiTBackbone):
    def __init__(self, **kwargs):
        super().__init__(model_name="mit_b5", **kwargs)


MIT_BACKBONES = {
    "mit_b0": mit_b0,
    "mit_b1": mit_b1,
    "mit_b2": mit_b2,
    "mit_b3": mit_b3,
    "mit_b4": mit_b4,
    "mit_b5": mit_b5,
}


def build_mit_backbone(model_name: str = "mit_b1", **kwargs) -> MiTBackbone:
    model_name = str(model_name).lower()
    if model_name not in MIT_BACKBONES:
        raise ValueError(
            f"Unknown MiT backbone '{model_name}'. Available: {list(MIT_BACKBONES.keys())}."
        )
    return MIT_BACKBONES[model_name](**kwargs)
