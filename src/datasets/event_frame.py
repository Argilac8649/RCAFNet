"""Shared event-frame construction helpers.

The dataset-specific modules only load their own event files.  All common
coordinate filtering, voxel-grid logic, and AET construction lives here to keep
DDD17 and DSEC behaviour consistent.
"""

from __future__ import annotations

import numpy as np


_FRAME_TYPE_CHANNELS = {
    '10c': 10,
    'aet': 6,
}


def frame_type_to_channels(frame_type: str) -> int:
    """Return the number of event channels produced by ``frame_type``."""
    frame_type = str(frame_type).lower()
    try:
        return _FRAME_TYPE_CHANNELS[frame_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported frame_type '{frame_type}'. "
            f"Expected one of {sorted(_FRAME_TYPE_CHANNELS)}."
        ) from exc


def _validate_shape(shape) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"shape must be (height, width), got {shape!r}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {(height, width)!r}")
    return height, width


def _as_events(events: np.ndarray) -> np.ndarray:
    events = np.asarray(events)
    if events.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"events must have shape (N, 4), got {events.shape!r}")
    return events


def normalize_voxel_grid(voxel_grid: np.ndarray, method: str = 'max') -> np.ndarray:
    """Normalize voxel grids channel-wise.

    Supported methods:
      - ``max``: divide by per-channel positive max and clip to [0, 1]
      - ``minmax``: per-channel min-max to [0, 1]
      - ``maxabs``/``signed_max``: divide by max absolute value, preserving sign
    """
    voxel_grid = np.asarray(voxel_grid, dtype=np.float32)
    if voxel_grid.size == 0:
        return voxel_grid

    method = str(method).lower()
    if method == 'max':
        max_vals = np.max(voxel_grid, axis=(1, 2), keepdims=True)
        max_vals = np.where(max_vals == 0, 1.0, max_vals)
        return np.clip(voxel_grid / max_vals, 0.0, 1.0).astype(np.float32, copy=False)

    if method == 'minmax':
        min_vals = np.min(voxel_grid, axis=(1, 2), keepdims=True)
        max_vals = np.max(voxel_grid, axis=(1, 2), keepdims=True)
        range_vals = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
        return np.clip((voxel_grid - min_vals) / range_vals, 0.0, 1.0).astype(np.float32, copy=False)

    if method in {'maxabs', 'signed_max'}:
        max_abs = np.max(np.abs(voxel_grid), axis=(1, 2), keepdims=True)
        max_abs = np.where(max_abs == 0, 1.0, max_abs)
        return (voxel_grid / max_abs).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported normalization method: {method}")


def _prepare_voxel_inputs(events: np.ndarray, shape):
    height, width = _validate_shape(shape)
    events = _as_events(events)
    if events.shape[0] == 0:
        return height, width, events

    xs = events[:, 0].astype(np.float32, copy=False)
    ys = events[:, 1].astype(np.float32, copy=False)
    pols = events[:, 2].astype(np.float32, copy=True)
    ts = events[:, 3].astype(np.float64, copy=False)

    if not np.all(ts[:-1] <= ts[1:]):
        order = np.argsort(ts)
        xs = xs[order]
        ys = ys[order]
        pols = pols[order]
        ts = ts[order]

    pols[pols == 0] = -1
    return height, width, (xs, ys, pols, ts)


def generate_voxel_grid(
    events: np.ndarray,
    shape,
    nr_temporal_bins: int = 5,
    separate_pol: bool = True,
    normalize: bool = True,
    normalize_method: str = 'minmax',
) -> np.ndarray:
    """Build a voxel grid with temporal interpolation and integer pixels."""
    assert nr_temporal_bins > 0
    height, width, prepared = _prepare_voxel_inputs(events, shape)
    out_channels = 2 * nr_temporal_bins if separate_pol else nr_temporal_bins
    if isinstance(prepared, np.ndarray):
        return np.zeros((out_channels, height, width), dtype=np.float32)

    xs_f, ys_f, pols, ts = prepared
    xs = np.floor(xs_f).astype(np.int32)
    ys = np.floor(ys_f).astype(np.int32)

    voxel_grid_positive = np.zeros((nr_temporal_bins, height, width), dtype=np.float32).ravel()
    voxel_grid_negative = np.zeros((nr_temporal_bins, height, width), dtype=np.float32).ravel()

    first_stamp = ts[0]
    last_stamp = ts[-1]
    delta_t = last_stamp - first_stamp
    if delta_t == 0:
        delta_t = 1.0
    ts_normalized = (nr_temporal_bins - 1) * (ts - first_stamp) / delta_t

    t0 = ts_normalized.astype(np.int32)
    dt = ts_normalized - t0
    vals_left = 1.0 - dt
    vals_right = dt
    pos_mask = pols > 0

    valid_pos = (
        (xs >= 0) & (xs < width)
        & (ys >= 0) & (ys < height)
        & (ts_normalized >= 0) & (ts_normalized < nr_temporal_bins)
    )

    def add_to(grid: np.ndarray, mask: np.ndarray, t: np.ndarray, weights: np.ndarray) -> None:
        valid = mask & valid_pos & (t >= 0) & (t < nr_temporal_bins)
        if not np.any(valid):
            return
        np.add.at(
            grid,
            xs[valid] + ys[valid] * width + t[valid] * width * height,
            weights[valid],
        )

    add_to(voxel_grid_positive, pos_mask, t0, vals_left)
    add_to(voxel_grid_positive, pos_mask, t0 + 1, vals_right)
    add_to(voxel_grid_negative, ~pos_mask, t0, vals_left)
    add_to(voxel_grid_negative, ~pos_mask, t0 + 1, vals_right)

    voxel_grid_positive = voxel_grid_positive.reshape((nr_temporal_bins, height, width))
    voxel_grid_negative = voxel_grid_negative.reshape((nr_temporal_bins, height, width))
    voxel_grid = (
        np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0)
        if separate_pol
        else voxel_grid_positive - voxel_grid_negative
    )

    if normalize:
        voxel_grid = normalize_voxel_grid(voxel_grid, method=normalize_method)
    return voxel_grid.astype(np.float32, copy=False)


def generate_voxel_grid_bilinear(
    events: np.ndarray,
    shape,
    nr_temporal_bins: int = 5,
    separate_pol: bool = True,
    normalize: bool = True,
    normalize_method: str = 'minmax',
) -> np.ndarray:
    """Build a voxel grid with temporal and spatial bilinear interpolation.

    ``separate_pol=False`` returns ``nr_temporal_bins`` signed channels where
    positive-polarity events add and negative-polarity events subtract.  This
    avoids the old bug where merged-polarity channels were treated like the
    separated positive/negative layout.
    """
    assert nr_temporal_bins > 0
    height, width, prepared = _prepare_voxel_inputs(events, shape)
    out_channels = 2 * nr_temporal_bins if separate_pol else nr_temporal_bins
    voxel_grid = np.zeros((out_channels, height, width), dtype=np.float32)
    if isinstance(prepared, np.ndarray):
        return voxel_grid

    xs, ys, pols, ts = prepared
    first_stamp = ts[0]
    last_stamp = ts[-1]
    delta_t = last_stamp - first_stamp
    if delta_t == 0:
        delta_t = 1.0
    ts_normalized = (nr_temporal_bins - 1) * (ts - first_stamp) / delta_t

    t0 = np.floor(ts_normalized).astype(np.int32)
    dt = ts_normalized - t0

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = xs - x0
    dy = ys - y0

    spatial_terms = (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y0, dx * (1.0 - dy)),
        (x1, y1, dx * dy),
    )
    temporal_terms = (
        (t0, 1.0 - dt),
        (t0 + 1, dt),
    )

    pos_mask = pols > 0

    def valid_mask(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (
            (x >= 0) & (x < width)
            & (y >= 0) & (y < height)
            & (t >= 0) & (t < nr_temporal_bins)
        )

    for x, y, spatial_weight in spatial_terms:
        for t, temporal_weight in temporal_terms:
            valid = valid_mask(x, y, t)
            if not np.any(valid):
                continue
            weights = (spatial_weight * temporal_weight).astype(np.float32, copy=False)

            if separate_pol:
                pos = valid & pos_mask
                if np.any(pos):
                    np.add.at(voxel_grid, (t[pos], y[pos], x[pos]), weights[pos] * np.abs(pols[pos]))

                neg = valid & ~pos_mask
                if np.any(neg):
                    np.add.at(
                        voxel_grid,
                        (nr_temporal_bins + t[neg], y[neg], x[neg]),
                        weights[neg] * np.abs(pols[neg]),
                    )
            else:
                np.add.at(voxel_grid, (t[valid], y[valid], x[valid]), weights[valid] * pols[valid])

    if normalize:
        voxel_grid = normalize_voxel_grid(voxel_grid, method=normalize_method)
    return voxel_grid.astype(np.float32, copy=False)


def generate_activity_enhanced_tensor(
    events: np.ndarray,
    shape,
    nr_temporal_bins: int = 3,
    normalize: bool = False,
    normalize_method: str = 'minmax',
) -> np.ndarray:
    """Build EISNet-style Activity-Enhanced Tensor.

    The input event layout is ``[x, y, polarity, t_rel]``.  The output is
    ``[voxel_grid, activity_map]`` with shape ``[2 * nr_temporal_bins, H, W]``:
    signed voxel channels are positive accumulation minus negative
    accumulation, while activity channels count event activity without
    polarity.  EISNet keeps this representation unnormalized by default.
    """
    assert nr_temporal_bins > 0
    height, width, prepared = _prepare_voxel_inputs(events, shape)
    if isinstance(prepared, np.ndarray):
        return np.zeros((nr_temporal_bins * 2, height, width), dtype=np.float32)

    xs_f, ys_f, pols, ts = prepared
    xs = np.floor(xs_f).astype(np.int32)
    ys = np.floor(ys_f).astype(np.int32)

    signed_grid = np.zeros((nr_temporal_bins, height, width), dtype=np.float32).ravel()
    activity_map = np.zeros((nr_temporal_bins, height, width), dtype=np.float32).ravel()

    first_stamp = ts[0]
    last_stamp = ts[-1]
    delta_t = last_stamp - first_stamp
    if delta_t == 0:
        delta_t = 1.0
    ts_normalized = (nr_temporal_bins - 1) * (ts - first_stamp) / delta_t

    t0 = ts_normalized.astype(np.int32)
    dt = ts_normalized - t0
    terms = (
        (t0, 1.0 - dt),
        (t0 + 1, dt),
    )
    valid_base = (
        (xs >= 0) & (xs < width)
        & (ys >= 0) & (ys < height)
        & (ts_normalized >= 0) & (ts_normalized < nr_temporal_bins)
    )

    for t, weight in terms:
        valid = valid_base & (t >= 0) & (t < nr_temporal_bins)
        if not np.any(valid):
            continue
        linear = xs[valid] + ys[valid] * width + t[valid] * width * height
        np.add.at(signed_grid, linear, weight[valid] * pols[valid])
        np.add.at(activity_map, linear, np.ceil(weight[valid]).astype(np.float32, copy=False))

    signed_grid = signed_grid.reshape((nr_temporal_bins, height, width))
    activity_map = activity_map.reshape((nr_temporal_bins, height, width))
    aet = np.concatenate((signed_grid, activity_map), axis=0)
    if normalize:
        aet = normalize_voxel_grid(aet, method=normalize_method)
    return aet.astype(np.float32, copy=False)


def build_event_frame_from_events(
    events: np.ndarray,
    ori_size=(256, 512),
    frame_type: str = '10c',
    use_bilinear_voxel: bool = False,
) -> np.ndarray:
    """Convert raw events to one of the supported event-frame layouts."""
    frame_type = str(frame_type).lower()
    frame_type_to_channels(frame_type)  # validates the option early

    if frame_type == '10c':
        builder = generate_voxel_grid_bilinear if use_bilinear_voxel else generate_voxel_grid
        event_frame = builder(
            events,
            ori_size,
            nr_temporal_bins=5,
            separate_pol=True,
            normalize=True,
            normalize_method='minmax',
        )
    elif frame_type == 'aet':
        event_frame = generate_activity_enhanced_tensor(
            events,
            ori_size,
            nr_temporal_bins=3,
            normalize=False,
        )
    else:  # guarded by frame_type_to_channels; kept for readability
        raise ValueError(f"Unsupported frame_type '{frame_type}'")

    return event_frame.astype(np.float32, copy=False)
