"""Model builders for RCAFNet experiments."""

from __future__ import annotations

import torch

from .decode_heads.mlp import DecoderHead, MLPConvHead
from .decode_heads.unet import UNetDecoder
from .encoders.carefnet import ConfigurableCrossRecalibNetEncoder
from .encoders.eisnet import ConfigurableEISNetEncoder
from .necks.upsample import Upsample
from .registry import get_model_builder, register_model
from .segmentors.segnet import SegNet


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _image_size_wh(config):
    image_size = _cfg_get(config, "img_size_wh", None)
    if image_size is None:
        raise KeyError("Config is missing required key 'img_size_wh'.")
    return image_size


def _encoder_caffm_cfg(config):
    return _cfg_get(config.encoder, "caffm", None)


def _encoder_cross_feature_cfg(config):
    return _cfg_get(config.encoder, "cross_feature_recalibration", None)


def build_decoder(in_channels, num_classes, config):
    decoder_cfg = _cfg_get(config, "decoder", None)
    decoder_type = str(_cfg_get(decoder_cfg, "type", "mlp_conv")).strip().lower()
    dropout_ratio = float(_cfg_get(decoder_cfg, "dropout_ratio", 0.1))
    align_corners = _to_bool(_cfg_get(decoder_cfg, "align_corners", False))

    if decoder_type in ("mlp", "decoderhead", "segformer"):
        return DecoderHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            embed_dim=int(_cfg_get(decoder_cfg, "embed_dim", 768)),
            align_corners=align_corners,
        )

    if decoder_type in ("mlp_conv", "mlpconv", "mlp-conv", "enhanced_mlp"):
        refine_channels = _cfg_get(decoder_cfg, "refine_channels", None)
        if refine_channels is not None:
            refine_channels = int(refine_channels)
        return MLPConvHead(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            embed_dim=int(_cfg_get(decoder_cfg, "embed_dim", 512)),
            refine_channels=refine_channels,
            refine_blocks=int(_cfg_get(decoder_cfg, "refine_blocks", 2)),
            use_depthwise=_to_bool(_cfg_get(decoder_cfg, "use_depthwise", True)),
            align_corners=align_corners,
        )

    if decoder_type in ("unet", "unet_lite", "unet-lite", "light_unet"):
        return UNetDecoder(
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_channels=_cfg_get(decoder_cfg, "decoder_channels", [192, 96, 48]),
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
            input_order=str(_cfg_get(decoder_cfg, "input_order", "high_to_low")),
            use_depthwise=_to_bool(_cfg_get(decoder_cfg, "use_depthwise", True)),
        )

    raise ValueError(
        f"Unknown decoder type '{decoder_type}'. Available: mlp, mlp_conv, unet."
    )


@register_model("rcafnet")
def build_rcafnet(config):
    image_size = _image_size_wh(config)
    rgb_in_chans = _cfg_get(
        config.encoder,
        "rgb_in_chans",
        _cfg_get(config, "image_channels", 3),
    )
    encoder = ConfigurableCrossRecalibNetEncoder(
        backbone=_cfg_get(config.encoder, "backbone", "mit_b1"),
        rgb_backbone=_cfg_get(config.encoder, "rgb_backbone", None),
        event_backbone=_cfg_get(config.encoder, "event_backbone", None),
        frame_type=config.frame_type,
        rgb_in_chans=int(rgb_in_chans),
        cross_feature_cfg=_encoder_cross_feature_cfg(config),
        caffm_cfg=_encoder_caffm_cfg(config),
        pretrained=_cfg_get(config.encoder, "pretrained", None),
        rgb_pretrained=_cfg_get(config.encoder, "rgb_pretrained", None),
        event_pretrained=_cfg_get(config.encoder, "event_pretrained", None),
        event_in_chans=_cfg_get(config.encoder, "event_in_chans", None),
        img_size=image_size,
    )
    decoder = build_decoder(encoder.out_channels, config.num_class, config)
    return SegNet(encoder, decoder, Upsample(image_size))


@register_model("esinet")
def build_esinet(config):
    image_size = _image_size_wh(config)
    rgb_in_chans = _cfg_get(
        config.encoder,
        "rgb_in_chans",
        _cfg_get(config, "image_channels", 3),
    )
    encoder = ConfigurableEISNetEncoder(
        backbone=_cfg_get(config.encoder, "backbone", "mit_b2"),
        rgb_backbone=_cfg_get(config.encoder, "rgb_backbone", None),
        event_backbone=_cfg_get(config.encoder, "event_backbone", "mit_b0"),
        frame_type=config.frame_type,
        rgb_in_chans=int(rgb_in_chans),
        event_in_chans=_cfg_get(config.encoder, "event_in_chans", None),
        aet_rep=_cfg_get(config.encoder, "aet_rep", config.frame_type == "aet"),
        aet_bins=int(_cfg_get(config.encoder, "aet_bins", 3)),
        img_size=image_size,
        pretrained=_cfg_get(config.encoder, "pretrained", None),
        rgb_pretrained=_cfg_get(config.encoder, "rgb_pretrained", None),
        event_pretrained=_cfg_get(config.encoder, "event_pretrained", None),
        pretrained_input_adapt=_cfg_get(config.encoder, "pretrained_input_adapt", True),
        mrfm_cfg=_cfg_get(config.encoder, "mrfm", None),
    )
    decoder = build_decoder(encoder.out_channels, config.num_class, config)
    return SegNet(encoder, decoder, Upsample(image_size))


def build_model(model_name: str, config, device: str | torch.device | None = None):
    model = get_model_builder(model_name)(config)
    if device is not None:
        model = model.to(device)
    elif torch.cuda.is_available():
        model = model.cuda()
    return model
