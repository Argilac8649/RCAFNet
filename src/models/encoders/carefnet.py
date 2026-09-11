"""CAREFNet RGB/Event MiT encoder.

CAREFNet keeps the same two-stream MiT backbone layout as the existing
RGB/Event models, but uses QAFRM for feature rectification and CAFFM for
stage-wise feature fusion.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..backbones.mit import (
    MiTBackbone,
    _has_pretrained,
    frame_type_to_event_in_chans,
)
from ..necks.carefnet_fusion import (
    AsymmetricCAFFM,
    CAFFM,
)
from ..necks.cross_fe import CrossFeatureRecalibration
from ..necks.feature_enhance import FeatureEnhance
from ..necks.quality_rectify import QAFRM


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _is_sequence_cfg(value) -> bool:
    return (
        value is not None
        and not isinstance(value, (str, bytes))
        and hasattr(value, "__iter__")
    )


def _as_stage_list(value, num_stages: int = 4, name: str = "value"):
    if _is_sequence_cfg(value):
        values = list(value)
        if len(values) == 1:
            values = values * num_stages
        elif len(values) != num_stages:
            raise ValueError(
                f"{name} should be a scalar or a list with {num_stages} values, "
                f"got {values!r}."
            )
        return values
    return [value for _ in range(num_stages)]


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


class IdentityStageRefiner(nn.Module):
    """No-op stage refiner used when feature refinement is disabled."""

    def __init__(self):
        super().__init__()
        self.debug_enabled = False
        self.debug_to_cpu = False
        self.debug_cache = {}

    @staticmethod
    def _debug_feature_map(x: torch.Tensor, to_cpu: bool = False):
        value = x.detach().abs().mean(dim=1, keepdim=True).float()
        return value.cpu() if to_cpu else value

    def set_debug(self, enabled: bool = True, to_cpu: bool = False):
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(to_cpu)
        self.debug_cache = {}
        return self

    def forward(self, rgb: torch.Tensor, event: torch.Tensor):
        if self.debug_enabled:
            self.debug_cache = {
                "rgb_before": self._debug_feature_map(rgb, self.debug_to_cpu),
                "event_before": self._debug_feature_map(event, self.debug_to_cpu),
                "rgb_after": self._debug_feature_map(rgb, self.debug_to_cpu),
                "event_after": self._debug_feature_map(event, self.debug_to_cpu),
                "identity": True,
            }
        return rgb, event


class RgbPassThroughFusion(nn.Module):
    """Return RGB features directly when CAFFM is disabled."""

    def __init__(self):
        super().__init__()
        self.debug_enabled = False
        self.debug_to_cpu = False
        self.debug_cache = {}

    @staticmethod
    def _debug_feature_map(x: torch.Tensor, to_cpu: bool = False):
        value = x.detach().abs().mean(dim=1, keepdim=True).float()
        return value.cpu() if to_cpu else value

    def set_debug(self, enabled: bool = True, to_cpu: bool = False):
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(to_cpu)
        self.debug_cache = {}
        return self

    def forward(self, rgb: torch.Tensor, event: torch.Tensor):
        if self.debug_enabled:
            self.debug_cache = {
                "rgb_before": self._debug_feature_map(rgb, self.debug_to_cpu),
                "event_before": self._debug_feature_map(event, self.debug_to_cpu),
                "output": self._debug_feature_map(rgb, self.debug_to_cpu),
                "identity": True,
                "fusion_type": "rgb_pass_through",
                "use_cross": False,
            }
        return rgb


class DualModalRecalibrationModule(nn.Module):
    """Dual-modal feature recalibration used by RecalibNet."""

    def __init__(
        self,
        dim_rgb: int,
        dim_event: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.dim_rgb = int(dim_rgb)
        self.dim_event = int(dim_event)
        if self.dim_rgb <= 0 or self.dim_event <= 0:
            raise ValueError(
                f"dim_rgb/dim_event should be positive, got "
                f"{dim_rgb}/{dim_event}."
            )
        self.enhance = FeatureEnhance(
            dim_event=self.dim_event,
            dim_rgb=self.dim_rgb,
            reduction=reduction,
            spatial_kernel_size=spatial_kernel_size,
        )
        self.debug_enabled = False
        self.debug_to_cpu = False
        self.debug_cache = {}

    @staticmethod
    def _debug_feature_map(x: torch.Tensor, to_cpu: bool = False):
        value = x.detach().abs().mean(dim=1, keepdim=True).float()
        return value.cpu() if to_cpu else value

    @staticmethod
    def _debug_detach(x: torch.Tensor, to_cpu: bool = False):
        value = x.detach().float()
        return value.cpu() if to_cpu else value

    def set_debug(self, enabled: bool = True, to_cpu: bool = False):
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(to_cpu)
        self.debug_cache = {}
        return self

    def forward(self, rgb: torch.Tensor, event: torch.Tensor):
        if rgb.ndim != 4 or event.ndim != 4:
            raise ValueError(
                f"DualModalRecalibrationModule expects NCHW tensors, got "
                f"rgb={tuple(rgb.shape)}, event={tuple(event.shape)}."
            )
        if rgb.shape[0] != event.shape[0] or rgb.shape[2:] != event.shape[2:]:
            raise ValueError(
                "DualModalRecalibrationModule expects matched batch/spatial shapes, "
                f"got rgb={tuple(rgb.shape)}, event={tuple(event.shape)}."
            )
        if rgb.shape[1] != self.dim_rgb or event.shape[1] != self.dim_event:
            raise ValueError(
                "DualModalRecalibrationModule input channels do not match config, "
                f"got rgb={rgb.shape[1]}, event={event.shape[1]}, "
                f"expected rgb={self.dim_rgb}, event={self.dim_event}."
            )

        event_out, rgb_out = self.enhance(event, rgb)
        if self.debug_enabled:
            self.debug_cache = {
                "rgb_before": self._debug_feature_map(rgb, self.debug_to_cpu),
                "event_before": self._debug_feature_map(event, self.debug_to_cpu),
                "rgb_after": self._debug_feature_map(rgb_out, self.debug_to_cpu),
                "event_after": self._debug_feature_map(event_out, self.debug_to_cpu),
                "gamma_event": self._debug_detach(
                    self.enhance.gamma_event,
                    self.debug_to_cpu,
                ),
                "gamma_rgb": self._debug_detach(
                    self.enhance.gamma_rgb,
                    self.debug_to_cpu,
                ),
                "module": "dual_modal_feature_recalibration",
            }
        return rgb_out, event_out


class CrossFeatureRecalibrationModule(nn.Module):
    """Bidirectional cross-feature recalibration used by CrossRecalibNet."""

    def __init__(
        self,
        dim_rgb: int,
        dim_event: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.dim_rgb = int(dim_rgb)
        self.dim_event = int(dim_event)
        if self.dim_rgb <= 0 or self.dim_event <= 0:
            raise ValueError(
                f"dim_rgb/dim_event should be positive, got "
                f"{dim_rgb}/{dim_event}."
            )
        self.recalibrate = CrossFeatureRecalibration(
            dim_event=self.dim_event,
            dim_rgb=self.dim_rgb,
            reduction=reduction,
            spatial_kernel_size=spatial_kernel_size,
        )
        self.debug_enabled = False
        self.debug_to_cpu = False
        self.debug_cache = {}

    @staticmethod
    def _debug_feature_map(x: torch.Tensor, to_cpu: bool = False):
        value = x.detach().abs().mean(dim=1, keepdim=True).float()
        return value.cpu() if to_cpu else value

    @staticmethod
    def _debug_detach(x: torch.Tensor, to_cpu: bool = False):
        value = x.detach().float()
        return value.cpu() if to_cpu else value

    def set_debug(self, enabled: bool = True, to_cpu: bool = False):
        self.debug_enabled = bool(enabled)
        self.debug_to_cpu = bool(to_cpu)
        self.debug_cache = {}
        return self

    def forward(self, rgb: torch.Tensor, event: torch.Tensor):
        if rgb.ndim != 4 or event.ndim != 4:
            raise ValueError(
                f"CrossFeatureRecalibrationModule expects NCHW tensors, got "
                f"rgb={tuple(rgb.shape)}, event={tuple(event.shape)}."
            )
        if rgb.shape[0] != event.shape[0] or rgb.shape[2:] != event.shape[2:]:
            raise ValueError(
                "CrossFeatureRecalibrationModule expects matched batch/spatial shapes, "
                f"got rgb={tuple(rgb.shape)}, event={tuple(event.shape)}."
            )
        if rgb.shape[1] != self.dim_rgb or event.shape[1] != self.dim_event:
            raise ValueError(
                "CrossFeatureRecalibrationModule input channels do not match config, "
                f"got rgb={rgb.shape[1]}, event={event.shape[1]}, "
                f"expected rgb={self.dim_rgb}, event={self.dim_event}."
            )

        event_out, rgb_out = self.recalibrate(event, rgb)
        if self.debug_enabled:
            self.debug_cache = {
                "rgb_before": self._debug_feature_map(rgb, self.debug_to_cpu),
                "event_before": self._debug_feature_map(event, self.debug_to_cpu),
                "rgb_after": self._debug_feature_map(rgb_out, self.debug_to_cpu),
                "event_after": self._debug_feature_map(event_out, self.debug_to_cpu),
                "gamma_event_to_rgb": self._debug_detach(
                    self.recalibrate.gamma_event_to_rgb,
                    self.debug_to_cpu,
                ),
                "gamma_rgb_to_event": self._debug_detach(
                    self.recalibrate.gamma_rgb_to_event,
                    self.debug_to_cpu,
                ),
                "module": "dual_modal_cross_feature_recalibration",
            }
        return rgb_out, event_out


class CAREFNetEncoder(nn.Module):
    """Dual-stream MiT encoder with QAFRM + CAFFM at each stage."""

    def __init__(
        self,
        backbone: str = "mit_b1",
        rgb_backbone: str | None = None,
        event_backbone: str | None = None,
        frame_type: str = "10c",
        rgb_in_chans: int = 3,
        event_in_chans: int | None = None,
        img_size=224,
        stage_refiner_cfg=None,
        caffm_cfg=None,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        norm_fuse=nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.rgb_backbone_name = str(rgb_backbone or backbone).lower()
        self.event_backbone_name = str(event_backbone or backbone).lower()
        self.backbone_name = self.rgb_backbone_name
        self.frame_type = frame_type
        self.rgb_in_chans = int(rgb_in_chans)
        self.event_in_chans = (
            frame_type_to_event_in_chans(frame_type)
            if event_in_chans is None
            else int(event_in_chans)
        )
        if self.rgb_in_chans <= 0:
            raise ValueError(
                f"rgb_in_chans should be positive, got {self.rgb_in_chans}."
            )
        if self.event_in_chans <= 0:
            raise ValueError(
                f"event_in_chans should be positive, got {self.event_in_chans}."
            )

        self.rgb_backbone = MiTBackbone(
            model_name=self.rgb_backbone_name,
            in_chans=self.rgb_in_chans,
            img_size=img_size,
            pretrained=None,
        )
        self.event_backbone = MiTBackbone(
            model_name=self.event_backbone_name,
            in_chans=self.event_in_chans,
            img_size=img_size,
            pretrained=None,
        )
        self.rgb_embed_dims = list(self.rgb_backbone.embed_dims)
        self.event_embed_dims = list(self.event_backbone.embed_dims)
        self._validate_stream_alignment()

        self.embed_dims = self.rgb_embed_dims
        self.num_heads = self.rgb_backbone.num_heads
        self.out_channels = self.embed_dims
        self.num_stages = len(self.embed_dims)
        self.visualization_enabled = False
        self.visualization_cache = {"stages": []}

        self.stage_refiners = self._build_stage_refiners(
            self.rgb_embed_dims,
            self.event_embed_dims,
            stage_refiner_cfg,
            self.num_stages,
        )

        self.caffms = self._build_caffms(
            self.rgb_embed_dims,
            self.event_embed_dims,
            self.num_heads,
            norm_fuse,
            caffm_cfg,
            self.num_stages,
        )

        rgb_pretrained = pretrained if rgb_pretrained is None else rgb_pretrained
        event_pretrained = pretrained if event_pretrained is None else event_pretrained
        self.init_weights(
            rgb_pretrained=rgb_pretrained,
            event_pretrained=event_pretrained,
        )

    def _validate_stream_alignment(self) -> None:
        if len(self.rgb_embed_dims) != len(self.event_embed_dims):
            raise ValueError(
                "RGB/Event MiT streams must have the same number of stages, "
                f"got rgb={len(self.rgb_embed_dims)}, "
                f"event={len(self.event_embed_dims)}."
            )

    @staticmethod
    def _load_stream_pretrained(
        stream_name: str,
        backbone: MiTBackbone,
        pretrained,
    ) -> None:
        if not _has_pretrained(pretrained):
            return
        if pretrained is True:
            raise ValueError(
                f"CAREFNetEncoder does not download pretrained weights for "
                f"{stream_name}. Pass a local checkpoint path instead."
            )
        backbone.init_weights(pretrained)

    def init_weights(
        self,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
    ) -> None:
        rgb_pretrained = pretrained if rgb_pretrained is None else rgb_pretrained
        event_pretrained = pretrained if event_pretrained is None else event_pretrained
        self._load_stream_pretrained("RGB stream", self.rgb_backbone, rgb_pretrained)
        self._load_stream_pretrained(
            "Event stream",
            self.event_backbone,
            event_pretrained,
        )

    def set_visualization(self, enabled: bool = True, debug_to_cpu: bool = False):
        self.visualization_enabled = bool(enabled)
        self.visualization_cache = {"stages": []}
        for module in list(self.stage_refiners) + list(self.caffms):
            if hasattr(module, "set_debug"):
                module.set_debug(enabled, to_cpu=debug_to_cpu)
        return self

    def get_visualization_cache(self):
        return self.visualization_cache

    @staticmethod
    def _build_stage_refiners(
        rgb_embed_dims: Sequence[int],
        event_embed_dims: Sequence[int],
        rectifier_cfg,
        num_stages: int,
    ) -> nn.ModuleList:
        stage_enabled = [
            _to_bool(value)
            for value in _as_stage_list(
                _cfg_get(rectifier_cfg, "enabled", True),
                num_stages=num_stages,
                name="encoder.rectifier.enabled",
            )
        ]
        if not any(stage_enabled):
            return nn.ModuleList([
                IdentityStageRefiner()
                for _ in range(num_stages)
            ])

        rectifier_type = str(_cfg_get(rectifier_cfg, "type", "qafrm")).strip().lower()
        if rectifier_type != "qafrm":
            raise ValueError(
                f"Unsupported encoder.rectifier.type='{rectifier_type}'. "
                "Expected 'qafrm'. Use encoder.rectifier.enabled=false to disable it."
            )

        spatial_hidden_dims = _as_stage_list(
            _cfg_get(rectifier_cfg, "spatial_hidden_dim", 8),
            num_stages=num_stages,
            name="encoder.rectifier.spatial_hidden_dim",
        )
        spatial_kernel_sizes = _as_stage_list(
            _cfg_get(rectifier_cfg, "spatial_kernel_size", 7),
            num_stages=num_stages,
            name="encoder.rectifier.spatial_kernel_size",
        )

        gamma_inits = _as_stage_list(
            _cfg_get(rectifier_cfg, "gamma_init", 0.0),
            num_stages=num_stages,
            name="encoder.rectifier.gamma_init",
        )
        bidirectional = _as_stage_list(
            _cfg_get(rectifier_cfg, "bidirectional", False),
            num_stages=num_stages,
            name="encoder.rectifier.bidirectional",
        )
        modules = []
        for i in range(num_stages):
            if stage_enabled[i]:
                modules.append(
                    QAFRM(
                        dim=int(rgb_embed_dims[i]),
                        dim_event=int(event_embed_dims[i]),
                        spatial_hidden_dim=int(spatial_hidden_dims[i]),
                        spatial_kernel_size=int(spatial_kernel_sizes[i]),
                        gamma_init=float(gamma_inits[i]),
                        bidirectional=_to_bool(bidirectional[i]),
                    )
                )
            else:
                modules.append(IdentityStageRefiner())
        return nn.ModuleList(modules)

    @staticmethod
    def _build_caffms(
        rgb_embed_dims: Sequence[int],
        event_embed_dims: Sequence[int],
        num_heads,
        norm_fuse,
        caffm_cfg,
        num_stages: int,
    ) -> nn.ModuleList:
        stage_enabled = [
            _to_bool(value)
            for value in _as_stage_list(
                _cfg_get(caffm_cfg, "enabled", True),
                num_stages=num_stages,
                name="encoder.caffm.enabled",
            )
        ]
        reductions = _as_stage_list(
            _cfg_get(caffm_cfg, "reduction", 4),
            num_stages=num_stages,
            name="encoder.caffm.reduction",
        )
        fusion_heads = _as_stage_list(
            _cfg_get(caffm_cfg, "num_heads", num_heads),
            num_stages=num_stages,
            name="encoder.caffm.num_heads",
        )
        use_cross = _as_stage_list(
            _cfg_get(caffm_cfg, "use_cross", True),
            num_stages=num_stages,
            name="encoder.caffm.use_cross",
        )
        gamma_inits = _as_stage_list(
            _cfg_get(caffm_cfg, "gamma_init", -3.0),
            num_stages=num_stages,
            name="encoder.caffm.gamma_init",
        )
        attn_drops = _as_stage_list(
            _cfg_get(caffm_cfg, "attn_drop", 0.0),
            num_stages=num_stages,
            name="encoder.caffm.attn_drop",
        )
        proj_drops = _as_stage_list(
            _cfg_get(caffm_cfg, "proj_drop", 0.0),
            num_stages=num_stages,
            name="encoder.caffm.proj_drop",
        )
        fusion_types = _as_stage_list(
            _cfg_get(caffm_cfg, "fusion_type", "concat"),
            num_stages=num_stages,
            name="encoder.caffm.fusion_type",
        )
        attention_types = _as_stage_list(
            _cfg_get(caffm_cfg, "attention_type", "efficient"),
            num_stages=num_stages,
            name="encoder.caffm.attention_type",
        )
        window_sizes = _as_stage_list(
            _cfg_get(caffm_cfg, "window_size", 7),
            num_stages=num_stages,
            name="encoder.caffm.window_size",
        )
        shift_sizes = _as_stage_list(
            _cfg_get(caffm_cfg, "shift_size", None),
            num_stages=num_stages,
            name="encoder.caffm.shift_size",
        )
        qkv_biases = _as_stage_list(
            _cfg_get(caffm_cfg, "qkv_bias", False),
            num_stages=num_stages,
            name="encoder.caffm.qkv_bias",
        )
        qk_scales = _as_stage_list(
            _cfg_get(caffm_cfg, "qk_scale", None),
            num_stages=num_stages,
            name="encoder.caffm.qk_scale",
        )

        fusion_gamma_cfg = _cfg_get(caffm_cfg, "fusion_gamma_init", None)
        if fusion_gamma_cfg is None:
            fusion_gamma_inits = gamma_inits
        else:
            fusion_gamma_inits = _as_stage_list(
                fusion_gamma_cfg,
                num_stages=num_stages,
                name="encoder.caffm.fusion_gamma_init",
            )

        modules = []
        for i in range(num_stages):
            if not stage_enabled[i]:
                modules.append(RgbPassThroughFusion())
                continue

            dim_rgb = int(rgb_embed_dims[i])
            dim_event = int(event_embed_dims[i])
            common_kwargs = {
                "reduction": int(reductions[i]),
                "num_heads": int(fusion_heads[i]),
                "norm_layer": norm_fuse,
                "use_cross": _to_bool(use_cross[i]),
                "gamma_init": float(gamma_inits[i]),
                "attn_drop": float(attn_drops[i]),
                "proj_drop": float(proj_drops[i]),
                "fusion_type": str(fusion_types[i]),
                "fusion_gamma_init": float(fusion_gamma_inits[i]),
            }
            caffm_kwargs = {
                **common_kwargs,
                "attention_type": str(attention_types[i]),
                "window_size": window_sizes[i],
                "shift_size": shift_sizes[i],
                "qkv_bias": _to_bool(qkv_biases[i]),
                "qk_scale": qk_scales[i],
            }
            if dim_rgb == dim_event:
                modules.append(CAFFM(dim=dim_rgb, **caffm_kwargs))
            else:
                modules.append(
                    AsymmetricCAFFM(
                        dim_rgb=dim_rgb,
                        dim_event=dim_event,
                        **caffm_kwargs,
                    )
                )
        return nn.ModuleList(modules)

    def forward(self, image: torch.Tensor, event: torch.Tensor):
        if image.ndim != 4:
            raise ValueError(
                f"Image input should be [B, C, H, W], got {tuple(image.shape)}."
            )
        if event.ndim != 4:
            raise ValueError(
                f"Event input should be [B, C, H, W], got {tuple(event.shape)}."
            )
        if image.shape[1] != self.rgb_in_chans:
            raise ValueError(
                f"CAREFNet encoder expects {self.rgb_in_chans} image channels, "
                f"got {image.shape[1]}."
            )
        if event.shape[1] != self.event_in_chans:
            raise ValueError(
                f"Event input channel mismatch: model expects {self.event_in_chans}, "
                f"but got {event.shape[1]}. Check config.frame_type / event_in_chans."
            )
        if image.shape[0] != event.shape[0] or image.shape[-2:] != event.shape[-2:]:
            raise ValueError(
                "Image/Event inputs should have the same batch and spatial size, "
                f"got image={tuple(image.shape)}, event={tuple(event.shape)}."
            )

        rgb = image
        event_feat = event
        outs = []
        if self.visualization_enabled:
            self.visualization_cache = {"stages": []}

        for stage_idx in range(self.num_stages):
            rgb = self.rgb_backbone.forward_stage(rgb, stage_idx)
            event_feat = self.event_backbone.forward_stage(event_feat, stage_idx)
            rgb, event_feat = self.stage_refiners[stage_idx](rgb, event_feat)
            fused = self.caffms[stage_idx](rgb, event_feat)
            outs.append(fused)

            if self.visualization_enabled:
                refiner_cache = dict(
                    getattr(self.stage_refiners[stage_idx], "debug_cache", {})
                )
                caffm_cache = dict(getattr(self.caffms[stage_idx], "debug_cache", {}))
                self.visualization_cache["stages"].append({
                    "stage": stage_idx + 1,
                    "stage_refiner": refiner_cache,
                    "caffm": caffm_cache,
                    "feature_shape": tuple(fused.shape),
                })

        return outs


class ConfigurableCAREFNetEncoder(nn.Module):
    """Project-config wrapper for CAREFNet QAFRM + CAFFM encoder."""

    def __init__(
        self,
        backbone="mit_b1",
        rgb_backbone=None,
        event_backbone=None,
        frame_type="10c",
        rgb_in_chans: int = 3,
        rectifier_cfg=None,
        caffm_cfg=None,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        event_in_chans: int | None = None,
        img_size=224,
    ) -> None:
        super().__init__()
        self.encoder = CAREFNetEncoder(
            backbone=backbone,
            rgb_backbone=rgb_backbone,
            event_backbone=event_backbone,
            frame_type=frame_type,
            rgb_in_chans=rgb_in_chans,
            event_in_chans=event_in_chans,
            img_size=img_size,
            stage_refiner_cfg=rectifier_cfg,
            caffm_cfg=caffm_cfg,
            pretrained=pretrained,
            rgb_pretrained=rgb_pretrained,
            event_pretrained=event_pretrained,
        )
        self.backbone_name = self.encoder.backbone_name
        self.rgb_backbone_name = self.encoder.rgb_backbone_name
        self.event_backbone_name = self.encoder.event_backbone_name
        self.frame_type = frame_type
        self.rgb_in_chans = self.encoder.rgb_in_chans
        self.event_in_chans = self.encoder.event_in_chans
        self.embed_dims = self.encoder.embed_dims
        self.num_heads = self.encoder.num_heads
        self.num_stages = self.encoder.num_stages
        self.out_channels = self.encoder.out_channels

    def set_visualization(self, enabled: bool = True, debug_to_cpu: bool = False):
        self.encoder.set_visualization(enabled, debug_to_cpu=debug_to_cpu)
        return self

    def get_visualization_cache(self):
        return self.encoder.get_visualization_cache()

    def forward(self, image: torch.Tensor, event: torch.Tensor):
        return self.encoder(image, event)


class RecalibNetEncoder(CAREFNetEncoder):
    """Dual-stream MiT encoder with FeatureEnhance + CAFFM at each stage."""

    def __init__(
        self,
        backbone: str = "mit_b1",
        rgb_backbone: str | None = None,
        event_backbone: str | None = None,
        frame_type: str = "10c",
        rgb_in_chans: int = 3,
        event_in_chans: int | None = None,
        img_size=224,
        feature_enhance_cfg=None,
        caffm_cfg=None,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        norm_fuse=nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            backbone=backbone,
            rgb_backbone=rgb_backbone,
            event_backbone=event_backbone,
            frame_type=frame_type,
            rgb_in_chans=rgb_in_chans,
            event_in_chans=event_in_chans,
            img_size=img_size,
            stage_refiner_cfg=feature_enhance_cfg,
            caffm_cfg=caffm_cfg,
            pretrained=pretrained,
            rgb_pretrained=rgb_pretrained,
            event_pretrained=event_pretrained,
            norm_fuse=norm_fuse,
        )

    @staticmethod
    def _build_stage_refiners(
        rgb_embed_dims: Sequence[int],
        event_embed_dims: Sequence[int],
        feature_cfg,
        num_stages: int,
    ) -> nn.ModuleList:
        stage_enabled = [
            _to_bool(value)
            for value in _as_stage_list(
                _cfg_get(feature_cfg, "enabled", True),
                num_stages=num_stages,
                name="encoder.feature_enhance.enabled",
            )
        ]
        if not any(stage_enabled):
            return nn.ModuleList([
                IdentityStageRefiner()
                for _ in range(num_stages)
            ])

        reductions = _as_stage_list(
            _cfg_get(feature_cfg, "reduction", 16),
            num_stages=num_stages,
            name="encoder.feature_enhance.reduction",
        )
        spatial_kernel_sizes = _as_stage_list(
            _cfg_get(feature_cfg, "spatial_kernel_size", 7),
            num_stages=num_stages,
            name="encoder.feature_enhance.spatial_kernel_size",
        )

        modules = []
        for i in range(num_stages):
            if stage_enabled[i]:
                modules.append(
                    DualModalRecalibrationModule(
                        dim_rgb=int(rgb_embed_dims[i]),
                        dim_event=int(event_embed_dims[i]),
                        reduction=int(reductions[i]),
                        spatial_kernel_size=int(spatial_kernel_sizes[i]),
                    )
                )
            else:
                modules.append(IdentityStageRefiner())
        return nn.ModuleList(modules)


class ConfigurableRecalibNetEncoder(nn.Module):
    """Project-config wrapper for RecalibNet FeatureEnhance + CAFFM encoder."""

    def __init__(
        self,
        backbone="mit_b1",
        rgb_backbone=None,
        event_backbone=None,
        frame_type="10c",
        rgb_in_chans: int = 3,
        feature_enhance_cfg=None,
        caffm_cfg=None,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        event_in_chans: int | None = None,
        img_size=224,
    ) -> None:
        super().__init__()
        self.encoder = RecalibNetEncoder(
            backbone=backbone,
            rgb_backbone=rgb_backbone,
            event_backbone=event_backbone,
            frame_type=frame_type,
            rgb_in_chans=rgb_in_chans,
            event_in_chans=event_in_chans,
            img_size=img_size,
            feature_enhance_cfg=feature_enhance_cfg,
            caffm_cfg=caffm_cfg,
            pretrained=pretrained,
            rgb_pretrained=rgb_pretrained,
            event_pretrained=event_pretrained,
        )
        self.backbone_name = self.encoder.backbone_name
        self.rgb_backbone_name = self.encoder.rgb_backbone_name
        self.event_backbone_name = self.encoder.event_backbone_name
        self.frame_type = frame_type
        self.rgb_in_chans = self.encoder.rgb_in_chans
        self.event_in_chans = self.encoder.event_in_chans
        self.embed_dims = self.encoder.embed_dims
        self.num_heads = self.encoder.num_heads
        self.num_stages = self.encoder.num_stages
        self.out_channels = self.encoder.out_channels

    def set_visualization(self, enabled: bool = True, debug_to_cpu: bool = False):
        self.encoder.set_visualization(enabled, debug_to_cpu=debug_to_cpu)
        return self

    def get_visualization_cache(self):
        return self.encoder.get_visualization_cache()

    def forward(self, image: torch.Tensor, event: torch.Tensor):
        return self.encoder(image, event)


class CrossRecalibNetEncoder(CAREFNetEncoder):
    """Dual-stream MiT encoder with CrossFeatureRecalibration + CAFFM."""

    def __init__(
        self,
        backbone: str = "mit_b1",
        rgb_backbone: str | None = None,
        event_backbone: str | None = None,
        frame_type: str = "10c",
        rgb_in_chans: int = 3,
        event_in_chans: int | None = None,
        img_size=224,
        cross_feature_cfg=None,
        caffm_cfg=None,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        norm_fuse=nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            backbone=backbone,
            rgb_backbone=rgb_backbone,
            event_backbone=event_backbone,
            frame_type=frame_type,
            rgb_in_chans=rgb_in_chans,
            event_in_chans=event_in_chans,
            img_size=img_size,
            stage_refiner_cfg=cross_feature_cfg,
            caffm_cfg=caffm_cfg,
            pretrained=pretrained,
            rgb_pretrained=rgb_pretrained,
            event_pretrained=event_pretrained,
            norm_fuse=norm_fuse,
        )

    @staticmethod
    def _build_stage_refiners(
        rgb_embed_dims: Sequence[int],
        event_embed_dims: Sequence[int],
        cross_feature_cfg,
        num_stages: int,
    ) -> nn.ModuleList:
        stage_enabled = [
            _to_bool(value)
            for value in _as_stage_list(
                _cfg_get(cross_feature_cfg, "enabled", True),
                num_stages=num_stages,
                name="encoder.cross_feature_recalibration.enabled",
            )
        ]
        if not any(stage_enabled):
            return nn.ModuleList([
                IdentityStageRefiner()
                for _ in range(num_stages)
            ])

        reductions = _as_stage_list(
            _cfg_get(cross_feature_cfg, "reduction", 16),
            num_stages=num_stages,
            name="encoder.cross_feature_recalibration.reduction",
        )
        spatial_kernel_sizes = _as_stage_list(
            _cfg_get(cross_feature_cfg, "spatial_kernel_size", 7),
            num_stages=num_stages,
            name="encoder.cross_feature_recalibration.spatial_kernel_size",
        )

        modules = []
        for i in range(num_stages):
            if stage_enabled[i]:
                modules.append(
                    CrossFeatureRecalibrationModule(
                        dim_rgb=int(rgb_embed_dims[i]),
                        dim_event=int(event_embed_dims[i]),
                        reduction=int(reductions[i]),
                        spatial_kernel_size=int(spatial_kernel_sizes[i]),
                    )
                )
            else:
                modules.append(IdentityStageRefiner())
        return nn.ModuleList(modules)


class ConfigurableCrossRecalibNetEncoder(nn.Module):
    """Project-config wrapper for CrossRecalibNet CrossFE + CAFFM encoder."""

    def __init__(
        self,
        backbone="mit_b1",
        rgb_backbone=None,
        event_backbone=None,
        frame_type="10c",
        rgb_in_chans: int = 3,
        cross_feature_cfg=None,
        caffm_cfg=None,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        event_in_chans: int | None = None,
        img_size=224,
    ) -> None:
        super().__init__()
        self.encoder = CrossRecalibNetEncoder(
            backbone=backbone,
            rgb_backbone=rgb_backbone,
            event_backbone=event_backbone,
            frame_type=frame_type,
            rgb_in_chans=rgb_in_chans,
            event_in_chans=event_in_chans,
            img_size=img_size,
            cross_feature_cfg=cross_feature_cfg,
            caffm_cfg=caffm_cfg,
            pretrained=pretrained,
            rgb_pretrained=rgb_pretrained,
            event_pretrained=event_pretrained,
        )
        self.backbone_name = self.encoder.backbone_name
        self.rgb_backbone_name = self.encoder.rgb_backbone_name
        self.event_backbone_name = self.encoder.event_backbone_name
        self.frame_type = frame_type
        self.rgb_in_chans = self.encoder.rgb_in_chans
        self.event_in_chans = self.encoder.event_in_chans
        self.embed_dims = self.encoder.embed_dims
        self.num_heads = self.encoder.num_heads
        self.num_stages = self.encoder.num_stages
        self.out_channels = self.encoder.out_channels

    def set_visualization(self, enabled: bool = True, debug_to_cpu: bool = False):
        self.encoder.set_visualization(enabled, debug_to_cpu=debug_to_cpu)
        return self

    def get_visualization_cache(self):
        return self.encoder.get_visualization_cache()

    def forward(self, image: torch.Tensor, event: torch.Tensor):
        return self.encoder(image, event)
