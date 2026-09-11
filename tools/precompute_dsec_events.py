from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load import load_config
from src.datasets.dsec.dataset import (
    PRECOMPUTED_METADATA_NAME,
    DsecDataset,
    build_precomputed_event_metadata,
    default_precomputed_event_root,
    write_precomputed_event_metadata,
)
from src.datasets.event_frame import frame_type_to_channels


def _cfg(config, name, default=None):
    if name in config:
        return config[name]
    return default


def _image_size_wh(config):
    if 'img_size_wh' not in config:
        raise KeyError("Config is missing required key 'img_size_wh'.")
    return tuple(int(v) for v in config.img_size_wh)


def _build_dataset(config, split):
    return DsecDataset(
        root_folder=_cfg(config, 'data_root'),
        image_size=_image_size_wh(config),
        frame_type=config.frame_type,
        mode=split,
        crop_bottom=int(_cfg(config, 'crop_bottom', 40)),
        use_rectify=bool(_cfg(config, 'use_rectify', True)),
        modality='rgb_event',
        normalize_image=bool(_cfg(config, 'normalize_image', True)),
        normalize_event=bool(_cfg(config, 'normalize_event', True)),
        use_precomputed_events=False,
    )


def _sample_event_paths(dataset, rgb_path):
    rgb_path = Path(rgb_path)
    frame_id = rgb_path.stem.split('_')[-1]
    seq_name = rgb_path.parent.parent.name
    event_path = rgb_path.parents[3] / 'event' / seq_name / 'data' / f'{frame_id}.npz'
    output_name = f'{frame_id}.npy'
    return seq_name, frame_id, event_path, output_name


def _check_metadata(output_root, metadata, overwrite):
    metadata_path = output_root / PRECOMPUTED_METADATA_NAME
    if not metadata_path.exists():
        return

    with metadata_path.open('r', encoding='utf-8') as handle:
        current = json.load(handle)
    if current == metadata:
        return
    if overwrite:
        return

    raise ValueError(
        f"Existing metadata differs from the requested cache settings: {metadata_path}. "
        "Pass --overwrite to replace it."
    )


def _save_event_frame(dataset, rgb_path, output_root, dtype, overwrite):
    seq_name, frame_id, event_path, output_name = _sample_event_paths(dataset, rgb_path)
    output_path = output_root / seq_name / 'data' / output_name
    if output_path.exists() and not overwrite:
        return 'skipped'

    event_frame = dataset.load_event(event_path)
    event_frame = event_frame.numpy().astype(dtype, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f'.{output_path.name}.tmp')
    with temp_path.open('wb') as handle:
        np.save(handle, event_frame)
    temp_path.replace(output_path)
    return 'written'


def precompute_dsec_events(args):
    config = load_config(args.config)
    if str(_cfg(config, 'train_dataset', 'dsec')).lower() != 'dsec':
        raise ValueError(f"Expected a DSEC config, got train_dataset={config.train_dataset!r}.")

    image_size = _image_size_wh(config)
    event_channels = frame_type_to_channels(str(config.frame_type).lower())

    output_root = Path(
        args.output_root
        or _cfg(config, 'precomputed_event_root', None)
        or default_precomputed_event_root(
            _cfg(config, 'data_root'),
            config.frame_type,
            bool(_cfg(config, 'use_rectify', True)),
            int(_cfg(config, 'crop_bottom', 40)),
            image_size,
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = build_precomputed_event_metadata(
        frame_type=config.frame_type,
        use_rectify=bool(_cfg(config, 'use_rectify', True)),
        crop_bottom=int(_cfg(config, 'crop_bottom', 40)),
        image_size=image_size,
        event_channels=event_channels,
        dtype=args.dtype,
    )
    _check_metadata(output_root, metadata, args.overwrite)
    write_precomputed_event_metadata(output_root, metadata)

    splits = ['train', 'val'] if args.split == 'all' else [args.split]
    total_written = 0
    total_skipped = 0
    started_at = time.time()

    for split in splits:
        dataset = _build_dataset(config, split)
        sample_paths = dataset.data_name
        if args.limit is not None:
            sample_paths = sample_paths[: int(args.limit)]

        print(f"Precomputing DSEC {split}: {len(sample_paths)} samples -> {output_root}", flush=True)
        for idx, rgb_path in enumerate(sample_paths, 1):
            status = _save_event_frame(
                dataset=dataset,
                rgb_path=rgb_path,
                output_root=output_root,
                dtype=np.dtype(args.dtype),
                overwrite=args.overwrite,
            )
            total_written += int(status == 'written')
            total_skipped += int(status == 'skipped')
            if idx == 1 or idx % args.log_interval == 0 or idx == len(sample_paths):
                elapsed = max(time.time() - started_at, 1e-6)
                rate = (total_written + total_skipped) / elapsed
                print(
                    f"  [{split}] {idx}/{len(sample_paths)} "
                    f"written={total_written} skipped={total_skipped} rate={rate:.2f} sample/s",
                    flush=True,
                )

    print(
        f"Done. written={total_written}, skipped={total_skipped}, "
        f"elapsed={(time.time() - started_at) / 60.0:.1f} min",
        flush=True,
    )
    print(f"Set use_precomputed_events: true and precomputed_event_root: {output_root}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute DSEC event frames as unnormalized .npy tensors.')
    parser.add_argument('--config', default='configs/rcafnet_dsec.yaml', help='DSEC experiment config path.')
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='all')
    parser.add_argument('--output-root', default=None, help='Override precomputed_event_root from config.')
    parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cached .npy files and metadata.')
    parser.add_argument('--limit', type=int, default=None, help='Debug: precompute only the first N samples per split.')
    parser.add_argument('--log-interval', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    precompute_dsec_events(parse_args())

