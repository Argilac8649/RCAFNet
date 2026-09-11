from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .utils import build_event_frame, train_sequences_namelist, val_sequences_namelist
from ..common import (
    load_dataset_statistics,
    modality_uses_event,
    modality_uses_image,
    normalize_modality,
    validate_dataset_statistics,
)
from ..event_frame import frame_type_to_channels


PRECOMPUTED_METADATA_NAME = 'metadata.json'
PRECOMPUTED_FORMAT_VERSION = 1


def default_precomputed_event_root(root_folder, frame_type, use_rectify, crop_bottom, image_size):
    coord_name = 'rectified' if use_rectify else 'raw'
    width, height = (int(v) for v in image_size)
    cache_name = f'{str(frame_type).lower()}_{coord_name}_crop{int(crop_bottom)}_{width}x{height}'
    return Path(root_folder) / 'precomputed_event' / cache_name


def build_precomputed_event_metadata(
    frame_type,
    use_rectify,
    crop_bottom,
    image_size,
    event_channels,
    dtype='float32',
):
    width, height = (int(v) for v in image_size)
    return {
        'format_version': PRECOMPUTED_FORMAT_VERSION,
        'dataset': 'dsec',
        'frame_type': str(frame_type).lower(),
        'use_rectify': bool(use_rectify),
        'crop_bottom': int(crop_bottom),
        'img_size_wh': [width, height],
        'event_channels': int(event_channels),
        'normalized': False,
        'dtype': str(dtype),
        'layout': 'sequence/data/frame_id.npy',
    }


def write_precomputed_event_metadata(output_root, metadata):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / PRECOMPUTED_METADATA_NAME
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write('\n')


class DsecDataset(Dataset):
    def __init__(
        self,
        root_folder='/home/cdw/dsec_data',
        image_size=(640, 440),
        frame_type='10c',
        mode='train',
        crop_bottom=40,
        use_rectify=True,
        modality='rgb_event',
        normalize_image=True,
        normalize_event=True,
        use_precomputed_events=False,
        precomputed_event_root=None,
    ):
        self.root_folder = Path(root_folder)
        self.image_size = tuple(int(v) for v in image_size)
        self.frame_type = str(frame_type).lower()
        self.mode = mode
        self.crop_bottom = int(crop_bottom)
        self.use_rectify = bool(use_rectify)
        self.modality = normalize_modality(modality)
        self.use_image = modality_uses_image(self.modality)
        self.use_event = modality_uses_event(self.modality)
        self.normalize_image = bool(normalize_image)
        self.normalize_event = bool(normalize_event)
        self.use_precomputed_events = bool(use_precomputed_events)
        if precomputed_event_root is None:
            self.precomputed_event_root = default_precomputed_event_root(
                self.root_folder,
                self.frame_type,
                self.use_rectify,
                self.crop_bottom,
                self.image_size,
            )
        else:
            self.precomputed_event_root = Path(precomputed_event_root)

        if self.crop_bottom < 0:
            raise ValueError(f"crop_bottom must be >= 0, got {self.crop_bottom}")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError(f"image_size must be (width, height), got {image_size!r}")

        event_channels = frame_type_to_channels(self.frame_type)
        self.event_channels = event_channels
        cfg = load_dataset_statistics('dsec', self.frame_type)
        validate_dataset_statistics(cfg, 'dsec', self.frame_type, event_channels)
        self.rgb_mean, self.rgb_std = cfg['image_mean'], cfg['image_std']
        self.evt_mean, self.evt_std = cfg['event_mean'], cfg['event_std']

        if self.use_event and self.use_precomputed_events:
            self._validate_precomputed_event_root()

        if self.mode == 'train':
            sequences_namelist = train_sequences_namelist
        elif self.mode == 'val':
            sequences_namelist = val_sequences_namelist
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        root = self.root_folder / 'image'
        if not root.exists():
            raise FileNotFoundError(
                f"DSEC image root does not exist: {root}. "
                "Check data_root in the active config file."
            )

        self.data_name = sorted(
            p
            for city in sorted(os.scandir(root), key=lambda e: e.name)
            if city.is_dir() and city.name in sequences_namelist
            for p in self._scan_seq(str(root), city.name)
        )
        if not self.data_name:
            raise RuntimeError(
                f"No DSEC samples found under {root} for mode='{self.mode}'. "
                f"Expected sequences: {sequences_namelist}."
            )

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        rgb_file = Path(self.data_name[index])
        rgb = self.load_image(rgb_file) if self.use_image else self.empty_image()

        frame_id = rgb_file.stem.split('_')[-1]
        seq_name = rgb_file.parent.parent.name
        event_file = rgb_file.parents[3] / 'event' / seq_name / 'data' / f'{frame_id}.npz'
        evt = self.load_event(event_file, seq_name, frame_id) if self.use_event else self.empty_event()

        semantic_file = rgb_file.parents[3] / 'label' / seq_name / '11classes' / f'{frame_id}.png'
        label = self.load_label(semantic_file)

        return rgb, evt, label

    def _crop_bottom_image(self, img: Image.Image, image_path: Path) -> Image.Image:
        if self.crop_bottom == 0:
            return img

        width, height = img.size
        if self.crop_bottom >= height:
            raise ValueError(
                f"crop_bottom={self.crop_bottom} is invalid for image '{image_path}' with height={height}"
            )
        return img.crop((0, 0, width, height - self.crop_bottom))

    def empty_image(self):
        """Return an RGB placeholder for event-only ablations."""
        return torch.zeros((3, self.image_size[1], self.image_size[0]), dtype=torch.float32)

    def empty_event(self):
        """Return an event placeholder for RGB-only ablations."""
        return torch.zeros(
            (self.event_channels, self.image_size[1], self.image_size[0]),
            dtype=torch.float32,
        )

    def load_image(self, image_path):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"DSEC image file does not exist: {image_path}")
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self._crop_bottom_image(img, Path(image_path))
                img = img.resize(self.image_size, Image.BILINEAR)
                return TF.to_tensor(img)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Cannot open image '{image_path}'") from e

    def load_event(self, event_path, seq_name=None, frame_id=None):
        if self.use_precomputed_events:
            if seq_name is None or frame_id is None:
                raise ValueError('seq_name and frame_id are required when use_precomputed_events=True.')
            return self.load_precomputed_event(seq_name, frame_id)

        if not Path(event_path).exists():
            raise FileNotFoundError(f"DSEC event file does not exist: {event_path}")

        event_frame = build_event_frame(
            events_data_path=event_path,
            use_rectify=self.use_rectify,
            frame_type=self.frame_type,
        )

        if self.crop_bottom > 0:
            if self.crop_bottom >= event_frame.shape[1]:
                raise ValueError(
                    f"crop_bottom={self.crop_bottom} is invalid for event frame '{event_path}' "
                    f"with height={event_frame.shape[1]}"
                )
            event_frame = event_frame[:, :-self.crop_bottom, :]

        event_frame = torch.from_numpy(event_frame)
        out = F.interpolate(
            event_frame[None, ...],
            size=self.image_size[::-1],
            mode='bilinear',
            align_corners=False,
        )
        return out[0]

    def load_precomputed_event(self, seq_name, frame_id):
        event_path = self.precomputed_event_root / str(seq_name) / 'data' / f'{frame_id}.npy'
        if not event_path.exists():
            raise FileNotFoundError(
                f"Precomputed DSEC event frame does not exist: {event_path}. "
                "Run tools/precompute_dsec_events.py or set use_precomputed_events=false."
            )

        event_frame = np.load(event_path)
        if event_frame.ndim != 3:
            raise ValueError(f"Precomputed event frame must be [C,H,W], got {event_frame.shape}: {event_path}")
        expected_shape = (self.event_channels, self.image_size[1], self.image_size[0])
        if tuple(event_frame.shape) != expected_shape:
            raise ValueError(
                f"Precomputed event frame shape mismatch for {event_path}: "
                f"expected {expected_shape}, got {tuple(event_frame.shape)}."
            )
        event_frame = event_frame.astype(np.float32, copy=False)
        return torch.from_numpy(event_frame)

    def _validate_precomputed_event_root(self):
        if not self.precomputed_event_root.exists():
            raise FileNotFoundError(
                f"Precomputed DSEC event root does not exist: {self.precomputed_event_root}. "
                "Run tools/precompute_dsec_events.py first or disable use_precomputed_events."
            )

        metadata_path = self.precomputed_event_root / PRECOMPUTED_METADATA_NAME
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed DSEC metadata: {metadata_path}. "
                "Regenerate the cache with tools/precompute_dsec_events.py."
            )

        with metadata_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)

        expected = build_precomputed_event_metadata(
            frame_type=self.frame_type,
            use_rectify=self.use_rectify,
            crop_bottom=self.crop_bottom,
            image_size=self.image_size,
            event_channels=self.event_channels,
        )
        checked_keys = (
            'format_version',
            'dataset',
            'frame_type',
            'use_rectify',
            'crop_bottom',
            'img_size_wh',
            'event_channels',
            'normalized',
            'layout',
        )
        mismatches = [
            f"{key}: expected {expected[key]!r}, got {metadata.get(key)!r}"
            for key in checked_keys
            if metadata.get(key) != expected[key]
        ]
        if mismatches:
            details = '; '.join(mismatches)
            raise ValueError(f"Precomputed DSEC metadata mismatch in {metadata_path}: {details}")

    def normalize(self, image, event):
        """Normalize image/event after all data augmentation."""
        if self.use_image and self.normalize_image:
            image = TF.normalize(image, self.rgb_mean, self.rgb_std)
        elif not self.use_image:
            image = torch.zeros_like(image)

        if self.use_event and self.normalize_event:
            event = TF.normalize(event, self.evt_mean, self.evt_std)
        elif not self.use_event:
            event = torch.zeros_like(event)
        return image, event

    def load_label(self, label_path):
        if not Path(label_path).exists():
            raise FileNotFoundError(f"DSEC label file does not exist: {label_path}")
        with Image.open(label_path) as label:
            label = label.resize(self.image_size, Image.NEAREST)
            label = np.asarray(label, dtype=np.uint8)
            return torch.tensor(label, dtype=torch.long)

    @lru_cache(maxsize=None)
    def _scan_seq(self, root, city):
        """Scan one DSEC sequence once and cache the result."""
        seq_path = Path(root) / city / 'evt_inf'
        try:
            return [
                str(seq_path / f.name)
                for f in os.scandir(seq_path)
                if f.is_file() and not f.name.startswith('.') and f.name.lower().endswith('.png')
            ]
        except FileNotFoundError:
            return []


if __name__ == '__main__':
    data_check = DsecDataset()
    print(len(data_check))
    image, event, labels = data_check[190]
    print(image.shape, image.dtype)
    print(event.shape, event.dtype)
    print(labels.shape, labels.dtype)
