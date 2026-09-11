"""DSEC dataset event helpers."""

from __future__ import annotations

import numpy as np

from ..event_frame import (
    build_event_frame_from_events,
)


train_sequences_namelist = [
    'zurich_city_00_a',
    'zurich_city_01_a',
    'zurich_city_02_a',
    'zurich_city_04_a',
    'zurich_city_05_a',
    'zurich_city_06_a',
    'zurich_city_07_a',
    'zurich_city_08_a',
]

val_sequences_namelist = [
    'zurich_city_13_a',
    'zurich_city_14_c',
    'zurich_city_15_a',
]


def _require_keys(data, file_name, keys):
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing keys {missing} in event file '{file_name}'")


def load_events(file_name, use_rectify=True):
    """Load DSEC events as ``[x, y, polarity, t_rel]``.

    Args:
        file_name: sliced event ``.npz`` path.
        use_rectify: use rectified coordinates (``x/y``) when true; otherwise
            use original coordinates (``x_o/y_o``).
    """
    x_key, y_key = ('x', 'y') if use_rectify else ('x_o', 'y_o')
    with np.load(file_name) as data:
        _require_keys(data, file_name, (x_key, y_key, 'p', 't'))
        t = data['t'].astype(np.float64, copy=False)
        t = t - t[0] if t.size else t
        return np.column_stack((data[x_key], data[y_key], data['p'], t))


def build_event_frame(events_data_path, use_rectify=True, ori_size=(480, 640), frame_type='10c'):
    events = load_events(events_data_path, use_rectify=use_rectify)
    return build_event_frame_from_events(
        events,
        ori_size=ori_size,
        frame_type=frame_type,
        use_bilinear_voxel=True,
    )
