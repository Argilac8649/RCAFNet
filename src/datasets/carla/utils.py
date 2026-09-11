"""CARLA dataset event helpers."""

from __future__ import annotations

import numpy as np

from ..event_frame import (
    build_event_frame_from_events,
    generate_voxel_grid,
    generate_voxel_grid_bilinear,
    normalize_voxel_grid,
)


def _require_keys(data, file_name, keys):
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing keys {missing} in event file '{file_name}'")


def load_events(file_name):
    """Load CARLA events as ``[x, y, polarity, t_rel]``.

    Coordinates are intentionally kept in their original signed/float dtype;
    boundary filtering is handled centrally in :mod:`fv_seg.datasets.event_frame`.
    """
    with np.load(file_name) as data:
        _require_keys(data, file_name, ('x', 'y', 'p', 't'))
        t = data['t'].astype(np.float64, copy=False)
        t = t - t[0] if t.size else t
        return np.column_stack((data['x'], data['y'], data['p'], t))


def build_event_frame(events_data_path, ori_size=(256, 512), frame_type='10c'):
    events = load_events(events_data_path)
    return build_event_frame_from_events(
        events,
        ori_size=ori_size,
        frame_type=frame_type,
        use_bilinear_voxel=False,
    )


ID2Color = {
    0:  (0,   0,   0),    # 背景
    1:  (70,  70,  70),   # 建筑物
    2:  (190, 153, 153),  # 栅栏
    3:  (250, 170, 160),  # 垃圾桶
    4:  (220, 20,  60),   # 人
    5:  (153, 153, 153),  # 杆子（电线杆）
    6:  (157, 234, 50),   # 车道线
    7:  (128, 64,  128),  # 路面（车道）
    8:  (244, 35,  232),  # 人行道
    9:  (107, 142, 35),   # 树
    10: (0,   0,   142),  # 汽车
    11: (102, 102, 156),  # 墙壁
    12: (220, 220, 0),    # 交通标志（信号灯）
}
