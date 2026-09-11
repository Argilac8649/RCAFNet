"""Shared dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_dataset_statistics(dataset_name: str, frame_type: str) -> Dict[str, Any]:
    """Load packaged normalization statistics for a dataset/frame type."""
    frame_type = str(frame_type).lower()
    stats_path = Path(__file__).resolve().parent / "statistics" / f"{dataset_name}_{frame_type}.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing statistics file: {stats_path}. "
            "Run tools/compute_mean_std.py or add the JSON file under datasets/statistics/."
        )
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_modality(modality: str | None) -> str:
    """Normalize user-facing modality aliases.

    Canonical values:
      - ``rgb_event``: load and use both RGB images and event frames
      - ``rgb``: load/use RGB only; event tensor is a zero placeholder
      - ``event``: load/use event only; RGB tensor is a zero placeholder
    """
    value = "rgb_event" if modality is None else str(modality).strip().lower()
    value = value.replace("-", "_").replace("+", "_")

    aliases = {
        "both": "rgb_event",
        "all": "rgb_event",
        "rgb_evt": "rgb_event",
        "rgb_events": "rgb_event",
        "image_event": "rgb_event",
        "img_event": "rgb_event",
        "image_events": "rgb_event",
        "img_events": "rgb_event",
        "rgb_event": "rgb_event",
        "rgb": "rgb",
        "image": "rgb",
        "img": "rgb",
        "rgb_only": "rgb",
        "image_only": "rgb",
        "img_only": "rgb",
        "event": "event",
        "events": "event",
        "evt": "event",
        "event_only": "event",
        "evt_only": "event",
    }

    if value not in aliases:
        raise ValueError(
            f"Unsupported modality '{modality}'. "
            "Expected one of: rgb_event, rgb, event."
        )
    return aliases[value]


def modality_uses_image(modality: str) -> bool:
    return normalize_modality(modality) in {"rgb", "rgb_event"}


def modality_uses_event(modality: str) -> bool:
    return normalize_modality(modality) in {"event", "rgb_event"}


def validate_dataset_statistics(
    stats: Dict[str, Any],
    dataset_name: str,
    frame_type: str,
    event_channels: int,
    image_channels: int = 3,
) -> None:
    """Validate normalization statistics before a training job starts."""
    required_keys = ("image_mean", "image_std", "event_mean", "event_std")
    missing = [key for key in required_keys if key not in stats]
    if missing:
        raise ValueError(
            f"{dataset_name}_{frame_type} statistics miss keys: {missing}. "
            "Please regenerate the statistics JSON."
        )

    image_channels = int(image_channels)
    image_stat_len = len(stats["image_mean"])
    image_std_len = len(stats["image_std"])
    image_stats_valid = (
        image_stat_len == image_channels
        and image_std_len == image_channels
    ) or (
        image_channels == 1
        and image_stat_len == 3
        and image_std_len == 3
    )
    if not image_stats_valid:
        raise ValueError(
            f"{dataset_name}_{frame_type} image statistics must contain "
            f"{image_channels} values, got mean={image_stat_len}, std={image_std_len}."
        )

    if len(stats["event_mean"]) != event_channels or len(stats["event_std"]) != event_channels:
        raise ValueError(
            f"{dataset_name}_{frame_type} event statistics must contain {event_channels} values, "
            f"got mean={len(stats['event_mean'])}, std={len(stats['event_std'])}."
        )
