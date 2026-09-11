"""EISNet encoder implemented in the CAREFNet training framework."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..backbones.mit import MiTBackbone, _has_pretrained, frame_type_to_event_in_chans
from ..necks.eisnet_fusion import AEIM, MRFM


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


class EISNetEncoder(nn.Module):
    """Original EISNet-style two-stream MiT encoder.

    Compared with CAREFNet, EISNet uses:
      - AEIM for AET event input before the event MiT stream
      - MRFM after each MiT stage for recalibration, cross-attention, and fusion
    """

    def __init__(
        self,
        backbone: str = "mit_b2",
        rgb_backbone: str | None = None,
        event_backbone: str | None = None,
        frame_type: str = "aet",
        rgb_in_chans: int = 3,
        event_in_chans: int | None = None,
        aet_rep: bool = True,
        aet_bins: int = 3,
        img_size=224,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        pretrained_input_adapt: bool = True,
        mrfm_cfg=None,
    ) -> None:
        super().__init__()

        self.rgb_backbone_name = str(rgb_backbone or backbone).lower()
        self.event_backbone_name = str(event_backbone or "mit_b0").lower()
        self.backbone_name = self.rgb_backbone_name
        self.frame_type = str(frame_type).lower()
        self.rgb_in_chans = int(rgb_in_chans)
        self.event_input_chans = (
            frame_type_to_event_in_chans(self.frame_type)
            if event_in_chans is None
            else int(event_in_chans)
        )
        self.aet_rep = _to_bool(aet_rep)
        self.aet_bins = int(aet_bins)
        self.pretrained_input_adapt = _to_bool(pretrained_input_adapt)

        if self.rgb_in_chans <= 0:
            raise ValueError(f"rgb_in_chans should be positive, got {self.rgb_in_chans}.")
        if self.event_input_chans <= 0:
            raise ValueError(
                f"event input channels should be positive, got {self.event_input_chans}."
            )
        if self.aet_rep:
            if self.event_input_chans != self.aet_bins * 2:
                raise ValueError(
                    "EISNet AET mode expects event channels = 2 * aet_bins, "
                    f"got event_in_chans={self.event_input_chans}, "
                    f"aet_bins={self.aet_bins}."
                )
            self.event_backbone_in_chans = self.aet_bins
        else:
            self.event_backbone_in_chans = self.event_input_chans

        self.rgb_backbone = MiTBackbone(
            model_name=self.rgb_backbone_name,
            in_chans=self.rgb_in_chans,
            img_size=img_size,
            pretrained=None,
        )
        self.event_backbone = MiTBackbone(
            model_name=self.event_backbone_name,
            in_chans=self.event_backbone_in_chans,
            img_size=img_size,
            pretrained=None,
        )

        self.rgb_embed_dims = list(self.rgb_backbone.embed_dims)
        self.event_embed_dims = list(self.event_backbone.embed_dims)
        if len(self.rgb_embed_dims) != len(self.event_embed_dims):
            raise ValueError(
                "EISNet RGB/Event backbones must have the same number of stages, "
                f"got rgb={len(self.rgb_embed_dims)}, event={len(self.event_embed_dims)}."
            )

        self.embed_dims = self.rgb_embed_dims
        self.num_heads = self.rgb_backbone.num_heads
        self.out_channels = self.embed_dims
        self.num_stages = len(self.embed_dims)

        self.aeim = (
            AEIM(in_dim=1, out_dim=self.event_embed_dims[0])
            if self.aet_rep
            else nn.Identity()
        )

        head_count = int(_cfg_get(mrfm_cfg, "head_count", 4))
        reduction = int(_cfg_get(mrfm_cfg, "reduction", 16))
        self.MRFMs = nn.ModuleList([
            MRFM(
                dim=(self.event_embed_dims[i], self.rgb_embed_dims[i]),
                head_count=head_count,
                reduction=reduction,
            )
            for i in range(self.num_stages)
        ])

        rgb_pretrained = pretrained if rgb_pretrained is None else rgb_pretrained
        event_pretrained = pretrained if event_pretrained is None else event_pretrained
        self.init_weights(rgb_pretrained=rgb_pretrained, event_pretrained=event_pretrained)

    @staticmethod
    def _load_stream_pretrained(
        stream_name: str,
        backbone: MiTBackbone,
        pretrained,
        pretrained_input_adapt: bool = True,
    ) -> None:
        if not _has_pretrained(pretrained):
            return
        if pretrained is True:
            raise ValueError(
                f"EISNetEncoder does not download pretrained weights for {stream_name}. "
                "Pass a local checkpoint path instead."
            )
        backbone.init_weights(pretrained, adapt_input_conv=pretrained_input_adapt)

    def init_weights(self, pretrained=None, rgb_pretrained=None, event_pretrained=None) -> None:
        rgb_pretrained = pretrained if rgb_pretrained is None else rgb_pretrained
        event_pretrained = pretrained if event_pretrained is None else event_pretrained
        self._load_stream_pretrained(
            "RGB stream",
            self.rgb_backbone,
            rgb_pretrained,
            pretrained_input_adapt=self.pretrained_input_adapt,
        )
        self._load_stream_pretrained(
            "Event stream",
            self.event_backbone,
            event_pretrained,
            pretrained_input_adapt=self.pretrained_input_adapt,
        )

    def _prepare_event_input(self, event: torch.Tensor) -> torch.Tensor:
        if self.aet_rep:
            ev = event[:, :self.aet_bins, :, :]
            activity = event[:, self.aet_bins:self.aet_bins * 2, :, :]
            return self.aeim(ev, activity)
        return event[:, :self.event_backbone_in_chans, :, :]

    def forward(self, image: torch.Tensor, event: torch.Tensor):
        if image.ndim != 4 or event.ndim != 4:
            raise ValueError(
                f"EISNet expects NCHW image/event tensors, got "
                f"image={tuple(image.shape)}, event={tuple(event.shape)}."
            )
        if image.shape[1] != self.rgb_in_chans:
            raise ValueError(
                f"EISNet expects {self.rgb_in_chans} image channels, got {image.shape[1]}."
            )
        if event.shape[1] != self.event_input_chans:
            raise ValueError(
                f"EISNet expects {self.event_input_chans} event channels, "
                f"got {event.shape[1]}."
            )
        if image.shape[0] != event.shape[0] or image.shape[-2:] != event.shape[-2:]:
            raise ValueError(
                "Image/Event inputs should have the same batch and spatial size, "
                f"got image={tuple(image.shape)}, event={tuple(event.shape)}."
            )

        rgb = image
        evt = self._prepare_event_input(event)
        outs = []
        for stage_idx in range(self.num_stages):
            evt = self.event_backbone.forward_stage(evt, stage_idx)
            rgb = self.rgb_backbone.forward_stage(rgb, stage_idx)
            fused, evt, rgb = self.MRFMs[stage_idx](evt, rgb)
            outs.append(fused)
        return outs


class ConfigurableEISNetEncoder(nn.Module):
    """Config wrapper for the EISNet encoder."""

    def __init__(
        self,
        backbone="mit_b2",
        rgb_backbone=None,
        event_backbone="mit_b0",
        frame_type="aet",
        rgb_in_chans=3,
        event_in_chans: int | None = None,
        aet_rep=True,
        aet_bins=3,
        img_size=224,
        pretrained=None,
        rgb_pretrained=None,
        event_pretrained=None,
        pretrained_input_adapt: bool = True,
        mrfm_cfg=None,
    ) -> None:
        super().__init__()
        self.encoder = EISNetEncoder(
            backbone=backbone,
            rgb_backbone=rgb_backbone,
            event_backbone=event_backbone,
            frame_type=frame_type,
            rgb_in_chans=rgb_in_chans,
            event_in_chans=event_in_chans,
            aet_rep=aet_rep,
            aet_bins=aet_bins,
            img_size=img_size,
            pretrained=pretrained,
            rgb_pretrained=rgb_pretrained,
            event_pretrained=event_pretrained,
            pretrained_input_adapt=pretrained_input_adapt,
            mrfm_cfg=mrfm_cfg,
        )
        self.backbone_name = self.encoder.backbone_name
        self.rgb_backbone_name = self.encoder.rgb_backbone_name
        self.event_backbone_name = self.encoder.event_backbone_name
        self.frame_type = self.encoder.frame_type
        self.rgb_in_chans = self.encoder.rgb_in_chans
        self.event_in_chans = self.encoder.event_input_chans
        self.embed_dims = self.encoder.embed_dims
        self.num_heads = self.encoder.num_heads
        self.num_stages = self.encoder.num_stages
        self.out_channels = self.encoder.out_channels

    def forward(self, image: torch.Tensor, event: torch.Tensor):
        return self.encoder(image, event)
