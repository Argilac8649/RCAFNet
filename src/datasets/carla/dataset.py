from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .utils import build_event_frame
from ..common import (
    load_dataset_statistics,
    modality_uses_event,
    modality_uses_image,
    normalize_modality,
    validate_dataset_statistics,
)
from ..event_frame import frame_type_to_channels


class CarlaDataset(Dataset):
    def __init__(
        self,
        root_folder='/media/cdw/WD_Disk/carla_data',
        towns_folder='Town01-03_train',
        image_size=(512, 256),
        frame_type='10c',
        modality='rgb_event',
    ):
        self.root_folder = Path(root_folder)
        self.towns_folder = towns_folder
        self.image_size = tuple(int(v) for v in image_size)
        self.frame_type = str(frame_type).lower()
        self.modality = normalize_modality(modality)
        self.use_image = modality_uses_image(self.modality)
        self.use_event = modality_uses_event(self.modality)

        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError(f"image_size must be (width, height), got {image_size!r}")

        event_channels = frame_type_to_channels(self.frame_type)
        self.event_channels = event_channels
        cfg = load_dataset_statistics('carla', self.frame_type)
        validate_dataset_statistics(cfg, 'carla', self.frame_type, event_channels)
        self.rgb_mean, self.rgb_std = cfg['image_mean'], cfg['image_std']
        self.evt_mean, self.evt_std = cfg['event_mean'], cfg['event_std']

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        root = self.root_folder / self.towns_folder
        if not root.exists():
            raise FileNotFoundError(
                f"CARLA split root does not exist: {root}. "
                "Check data_root/train_folder/val_folder in the active config file."
            )

        self.data_name = sorted(
            p
            for town in sorted(os.scandir(root), key=lambda e: e.name)
            if town.is_dir()
            for seq in sorted(os.scandir(town.path), key=lambda e: e.name)
            if seq.is_dir()
            for p in self._scan_seq(str(root), town.name, seq.name)
        )
        if not self.data_name:
            raise RuntimeError(f"No CARLA samples found under {root}.")

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        rgb_file = Path(self.data_name[index])
        rgb = self.load_image(rgb_file) if self.use_image else self.empty_image()

        event_file = rgb_file.parents[2] / 'events' / 'data' / f'{rgb_file.stem.replace("_image", "_events")}.npz'
        evt = self.load_event(event_file) if self.use_event else self.empty_event()

        semantic_file = rgb_file.parents[2] / 'semantic' / 'data' / f'{rgb_file.stem.replace("_image", "_gt_labelIds")}.png'
        label = self.load_label(semantic_file)

        return rgb, evt, label

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
            raise FileNotFoundError(f"CARLA image file does not exist: {image_path}")
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize(self.image_size, Image.BILINEAR)
                return TF.to_tensor(img)
        except (IOError, OSError) as err:
            raise RuntimeError(f"Cannot open or process image '{image_path}': {err}") from err

    def load_event(self, event_path):
        if not Path(event_path).exists():
            raise FileNotFoundError(f"CARLA event file does not exist: {event_path}")

        event_frame = build_event_frame(events_data_path=event_path, frame_type=self.frame_type)
        event_frame = torch.from_numpy(event_frame)
        out = F.interpolate(
            event_frame[None, ...],
            size=self.image_size[::-1],
            mode='bilinear',
            align_corners=False,
        )
        return out[0]

    def normalize(self, image, event):
        """Normalize image/event after all data augmentation."""
        if self.use_image:
            image = TF.normalize(image, self.rgb_mean, self.rgb_std)
        else:
            image = torch.zeros_like(image)

        if self.use_event:
            event = TF.normalize(event, self.evt_mean, self.evt_std)
        else:
            event = torch.zeros_like(event)
        return image, event

    def load_label(self, label_path):
        if not Path(label_path).exists():
            raise FileNotFoundError(f"CARLA label file does not exist: {label_path}")
        with Image.open(label_path) as label:
            label = label.resize(self.image_size, Image.NEAREST)
            label = np.asarray(label, dtype=np.uint8)
            label = torch.tensor(label, dtype=torch.long)

            # CARLA labels: 0 background -> ignore, 1..N -> 0..N-1.
            label[label == 0] = 255
            label[label != 255] -= 1
            return label

    @lru_cache(maxsize=None)
    def _scan_seq(self, root, town, seq):
        """Scan one CARLA sequence once and cache the result."""
        seq_path = Path(root) / town / seq / 'rgb' / 'data'
        try:
            return [
                str(seq_path / f.name)
                for f in os.scandir(seq_path)
                if f.is_file() and not f.name.startswith('.') and f.name.lower().endswith('.png')
            ]
        except FileNotFoundError:
            return []


if __name__ == '__main__':
    data_check = CarlaDataset()
    print(len(data_check))
    image, event, labels = data_check[190]
    print(image.shape, image.dtype)
    print(event.shape, event.dtype)
    print(labels.shape, labels.dtype)
