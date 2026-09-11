"""DDD17 semantic-segmentation event helpers.

The local ``ddd17_seg`` layout is expected to be either::

    <root>/data/dir*/imgs
    <root>/data/dir*/segmentation_masks
    <root>/data/dir*/index/index_{10,50,250}ms.npy
    <root>/data/dir*/events.dat.t
    <root>/data/dir*/events.dat.xyp

or the ``data`` directory can be passed directly as ``root``.  Raw DDD17
events are stored as two memmap-friendly binary files:

* ``events.dat.t``: int64 timestamps
* ``events.dat.xyp``: int16/uint16 triples ``[x, y, polarity]``
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ..event_frame import build_event_frame_from_events


# ESS (uzh-rpg/ess) uses sorted DDD17 directories and splits them as:
# train = dirs[0], dirs[2], dirs[3], dirs[5], dirs[6]
# valid = dirs[1]
# test  = dirs[4]
# For the local ddd17_seg folders sorted as
# (dir0, dir1, dir3, dir4, dir5, dir6, dir7), this becomes:
DEFAULT_TRAIN_DIRS = ("dir0", "dir3", "dir4", "dir6", "dir7")
DEFAULT_VAL_DIRS = ("dir1",)
DEFAULT_TEST_DIRS = ("dir5",)

DDD17_ID2NAME = {
    0: "flat",
    1: "construction+sky",
    2: "object",
    3: "nature",
    4: "human",
    5: "vehicle",
    255: "ignore_labels",
}

DDD17_ID2COLOR = {
    0: (128, 64, 128),    # flat, Cityscapes road
    1: (70, 70, 70),      # construction+sky, Cityscapes building
    2: (153, 153, 153),   # object, Cityscapes pole/object gray
    3: (107, 142, 35),    # nature, Cityscapes vegetation
    4: (220, 20, 60),     # human, Cityscapes person
    5: (0, 0, 142),       # vehicle, Cityscapes car
    255: (0, 0, 0),       # ignore_labels
}

MASK_RE = re.compile(r"segmentation_(\d+)\.png$")


def resolve_data_root(root_folder: str | os.PathLike) -> Path:
    """Return the directory containing ``dir*`` sequence folders."""
    root = Path(root_folder)
    data_root = root / "data"
    if data_root.exists():
        return data_root
    return root


def normalize_sequence_list(sequences, default) -> tuple[str, ...]:
    """Convert YAML/OmegaConf/Python sequence lists to a tuple of strings."""
    if sequences is None:
        return tuple(default)
    if isinstance(sequences, str):
        return (sequences,)
    return tuple(str(seq) for seq in sequences)


def frame_id_from_mask_name(name: str) -> int | None:
    """Extract the 1-based frame id from ``segmentation_00001234.png``."""
    match = MASK_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def image_candidates(seq_dir: Path, frame_id: int) -> tuple[Path, ...]:
    """Return possible DDD17 image names for a mask frame id.

    Some folders use ``img_00000002.png`` while others use
    ``0000001489.png``.  Keep all observed patterns here so the Dataset does
    not need per-sequence special-casing.
    """
    imgs_dir = seq_dir / "imgs"
    return (
        imgs_dir / f"img_{frame_id:08d}.png",
        imgs_dir / f"{frame_id:010d}.png",
        imgs_dir / f"{frame_id:08d}.png",
    )


def find_image_file(seq_dir: Path, frame_id: int) -> Path | None:
    for candidate in image_candidates(seq_dir, frame_id):
        if candidate.exists():
            return candidate
    return None


def index_file_name(t_interval: int) -> str:
    t_interval = int(t_interval)
    if t_interval not in {10, 50, 250}:
        raise ValueError(f"DDD17 t_interval must be one of 10, 50, 250 ms, got {t_interval}")
    return f"index_{t_interval}ms.npy"


def open_event_memmaps(seq_dir: Path):
    """Open DDD17 index/timestamp/xyp files without loading all events."""
    if not (seq_dir / "index").exists():
        raise FileNotFoundError(f"DDD17 index directory does not exist: {seq_dir / 'index'}")

    t_file = seq_dir / "events.dat.t"
    xyp_file = seq_dir / "events.dat.xyp"
    if not t_file.exists():
        raise FileNotFoundError(f"DDD17 timestamp file does not exist: {t_file}")
    if not xyp_file.exists():
        raise FileNotFoundError(f"DDD17 xyp file does not exist: {xyp_file}")

    num_events = os.path.getsize(t_file) // 8
    t_events = np.memmap(t_file, dtype="int64", mode="r", shape=(num_events, 1))
    xyp_events = np.memmap(xyp_file, dtype="int16", mode="r", shape=(num_events, 3))
    return t_events, xyp_events


def load_index(seq_dir: Path, t_interval: int):
    index_path = seq_dir / "index" / index_file_name(t_interval)
    if not index_path.exists():
        raise FileNotFoundError(f"DDD17 index file does not exist: {index_path}")
    return np.load(index_path, mmap_mode="r")


def make_relative_events(xyp: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Build ``[x, y, polarity, t_rel]`` event array from raw slices."""
    if xyp.size == 0 or timestamps.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    xyp = np.asarray(xyp, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    timestamps = timestamps - timestamps[0]
    return np.column_stack((xyp, timestamps))


def build_event_frame(
    events: np.ndarray,
    ori_size=(260, 346),
    frame_type="10c",
    use_bilinear_voxel: bool = False,
) -> np.ndarray:
    """Convert DDD17 ``[x, y, polarity, t_rel]`` events to an event frame."""
    return build_event_frame_from_events(
        events,
        ori_size=ori_size,
        frame_type=frame_type,
        use_bilinear_voxel=use_bilinear_voxel,
    )
