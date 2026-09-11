"""Stable visualization helpers shared by scripts."""

from __future__ import annotations

import numpy as np


IGNORE_COLOR = (0, 0, 0)

SEMANTIC_COLORS = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic_sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "vehicle": (0, 0, 142),
    "object": (153, 153, 153),
    "trash_can": (120, 120, 120),
    "lane_line": (255, 255, 255),
}

DSEC_CLASS_NAMES = (
    "sky",
    "building",
    "fence",
    "person",
    "pole",
    "road",
    "sidewalk",
    "vegetation",
    "vehicle",
    "wall",
    "traffic_sign",
)

DDD17_CLASS_NAMES = (
    "flat",
    "construction+sky",
    "object",
    "nature",
    "human",
    "vehicle",
)

CARLA_CLASS_NAMES = (
    "building",
    "fence",
    "trash_can",
    "person",
    "pole",
    "lane_line",
    "road",
    "sidewalk",
    "vegetation",
    "vehicle",
    "wall",
    "traffic_sign",
)

DATASET_CLASS_NAMES = {
    "dsec": DSEC_CLASS_NAMES,
    "ddd17": DDD17_CLASS_NAMES,
    "carla": CARLA_CLASS_NAMES,
}

DATASET_PALETTES = {
    "dsec": np.array([SEMANTIC_COLORS[name] for name in DSEC_CLASS_NAMES], dtype=np.uint8),
    "ddd17": np.array([
        SEMANTIC_COLORS["road"],        # flat
        SEMANTIC_COLORS["building"],    # construction+sky
        SEMANTIC_COLORS["object"],      # object
        SEMANTIC_COLORS["vegetation"],  # nature
        SEMANTIC_COLORS["person"],      # human
        SEMANTIC_COLORS["vehicle"],     # vehicle
    ], dtype=np.uint8),
    "carla": np.array([SEMANTIC_COLORS[name] for name in CARLA_CLASS_NAMES], dtype=np.uint8),
}


def fallback_palette(num_classes: int, offset: int = 0) -> np.ndarray:
    """Generate deterministic colors for unknown datasets/classes."""
    palette = np.zeros((int(num_classes), 3), dtype=np.uint8)
    for i in range(palette.shape[0]):
        idx = i + int(offset)
        palette[i] = [
            (37 * idx + 128) % 255,
            (17 * idx + 64) % 255,
            (97 * idx + 32) % 255,
        ]
    return palette


def get_palette(dataset_name: str | None = None, num_classes: int | None = None) -> np.ndarray:
    """Return a dataset palette, extending it deterministically if needed."""
    key = str(dataset_name).lower() if dataset_name is not None else ""
    if key in DATASET_PALETTES:
        palette = DATASET_PALETTES[key].copy()
    elif num_classes is not None:
        palette = fallback_palette(int(num_classes))
    else:
        raise ValueError("Either a known dataset_name or num_classes must be provided.")

    if num_classes is not None:
        num_classes = int(num_classes)
        if num_classes < len(palette):
            palette = palette[:num_classes]
        elif num_classes > len(palette):
            extra = fallback_palette(num_classes - len(palette), offset=len(palette))
            palette = np.concatenate([palette, extra], axis=0)
    return palette


def colorize_label(
    label: np.ndarray,
    dataset_name: str | None = None,
    num_classes: int | None = None,
    ignore_index: int = 255,
) -> np.ndarray:
    """Convert a label map to an RGB image in ``[0, 1]``.

    The same semantic category keeps the same color across DDD17 and DSEC.
    """
    palette = get_palette(dataset_name=dataset_name, num_classes=num_classes)
    label = np.asarray(label, dtype=np.int64)

    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    valid = (label >= 0) & (label < len(palette)) & (label != int(ignore_index))
    out[valid] = palette[label[valid]]
    out[label == int(ignore_index)] = np.array(IGNORE_COLOR, dtype=np.uint8)
    return out.astype(np.float32) / 255.0
