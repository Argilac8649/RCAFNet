from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .utils import (
    DEFAULT_TEST_DIRS,
    DEFAULT_TRAIN_DIRS,
    DEFAULT_VAL_DIRS,
    build_event_frame,
    find_image_file,
    frame_id_from_mask_name,
    load_index,
    make_relative_events,
    normalize_sequence_list,
    open_event_memmaps,
    resolve_data_root,
)
from ..common import (
    load_dataset_statistics,
    modality_uses_event,
    modality_uses_image,
    normalize_modality,
    validate_dataset_statistics,
)
from ..event_frame import frame_type_to_channels


class DDD17Dataset(Dataset):
    """DDD17 RGB/event semantic segmentation dataset.

    Expected sample alignment:
      - grayscale image: ``dir*/imgs/img_00000002.png`` or ``dir*/imgs/0000000002.png``
      - label: ``dir*/segmentation_masks/segmentation_00000002.png``
      - event slice: ``dir*/index/index_{t_interval}ms.npy`` + raw event memmaps

    DDD17 provides grayscale frames.  CAREFNet keeps them as one luminance
    channel by default, matching the EISNet protocol.  MiT patch-embedding
    weights are adapted from RGB pretrained checkpoints when needed.

    Labels in the local ``ddd17_seg`` split are 6-class masks:
    ``0=flat, 1=construction+sky, 2=object, 3=nature, 4=human,
    5=vehicle``; ``255`` is ignored.
    """

    def __init__(
        self,
        root_folder="/media/cdw/WD_Disk/ddd17/ddd17_seg",
        image_size=(346, 200),
        frame_type="10c",
        mode="train",
        crop_bottom=60,
        t_interval=50,
        ori_size=(260, 346),
        train_dirs=None,
        val_dirs=None,
        test_dirs=None,
        modality="rgb_event",
        force_grayscale=True,
        image_channels=1,
        use_precomputed_events=False,
        use_bilinear_voxel=False,
        event_slice_mode="inclusive",
        normalize_image=True,
        normalize_event=True,
    ):
        self.root_folder = Path(root_folder)
        self.data_root = resolve_data_root(root_folder)
        self.image_size = tuple(int(v) for v in image_size)
        self.frame_type = str(frame_type).lower()
        self.mode = str(mode).lower()
        self.crop_bottom = int(crop_bottom)
        self.t_interval = int(t_interval)
        self.ori_size = tuple(int(v) for v in ori_size)
        self.modality = normalize_modality(modality)
        self.use_image = modality_uses_image(self.modality)
        self.use_event = modality_uses_event(self.modality)
        self.force_grayscale = bool(force_grayscale)
        self.image_channels = int(image_channels)
        self.use_precomputed_events = bool(use_precomputed_events)
        self.use_bilinear_voxel = bool(use_bilinear_voxel)
        self.event_slice_mode = str(event_slice_mode).strip().lower()
        self.normalize_image = bool(normalize_image)
        self.normalize_event = bool(normalize_event)

        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError(f"image_size must be (width, height), got {image_size!r}")
        if len(self.ori_size) != 2 or min(self.ori_size) <= 0:
            raise ValueError(f"ori_size must be (height, width), got {ori_size!r}")
        if self.crop_bottom < 0:
            raise ValueError(f"crop_bottom must be >= 0, got {self.crop_bottom}")
        if self.crop_bottom >= self.ori_size[0]:
            raise ValueError(
                f"crop_bottom={self.crop_bottom} is invalid for ori_size={self.ori_size}"
            )
        if self.image_channels not in {1, 3}:
            raise ValueError(f"image_channels must be 1 or 3, got {self.image_channels}.")
        if self.event_slice_mode not in {"inclusive", "exclusive", "original", "eisnet"}:
            raise ValueError(
                "event_slice_mode must be one of: inclusive, exclusive, original, eisnet; "
                f"got {event_slice_mode!r}."
            )

        event_channels = frame_type_to_channels(self.frame_type)
        self.event_channels = event_channels
        cfg = load_dataset_statistics("ddd17", self.frame_type)
        validate_dataset_statistics(
            cfg,
            "ddd17",
            self.frame_type,
            event_channels,
            image_channels=self.image_channels,
        )
        self.rgb_mean = cfg["image_mean"][:self.image_channels]
        self.rgb_std = cfg["image_std"][:self.image_channels]
        self.evt_mean, self.evt_std = cfg["event_mean"], cfg["event_std"]

        if self.mode == "train":
            sequences = normalize_sequence_list(train_dirs, DEFAULT_TRAIN_DIRS)
        elif self.mode == "val":
            sequences = normalize_sequence_list(val_dirs, DEFAULT_VAL_DIRS)
        elif self.mode == "test":
            sequences = normalize_sequence_list(test_dirs, DEFAULT_TEST_DIRS)
        elif self.mode in {"all", "full"}:
            sequences = tuple(
                sorted(
                    p.name
                    for p in self.data_root.iterdir()
                    if p.is_dir() and p.name.startswith("dir")
                )
            )
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")
        self.sequences = sequences

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"DDD17 data root does not exist: {self.data_root}. "
                "Check data_root in the active config file."
            )

        self.data_name = []
        missing_images = 0
        for seq_name in self.sequences:
            seq_dir = self.data_root / seq_name
            if not seq_dir.exists():
                warnings.warn(f"DDD17 sequence directory does not exist and will be skipped: {seq_dir}")
                continue
            seq_samples, seq_missing = self._scan_seq(str(seq_dir))
            self.data_name.extend(seq_samples)
            missing_images += seq_missing

        self.data_name = sorted(self.data_name, key=lambda item: (item[2], item[3]))
        if missing_images:
            warnings.warn(f"Skipped {missing_images} DDD17 masks because the paired RGB image was missing.")
        if not self.data_name:
            raise RuntimeError(
                f"No DDD17 samples found under {self.data_root} for mode='{self.mode}'. "
                f"Expected sequences: {self.sequences}."
            )

        # Lazily opened per process.  This avoids pickling memmap handles when
        # DataLoader uses worker processes.
        self._event_cache = {}

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        rgb_file, label_file, seq_name, frame_id = self.data_name[index]
        rgb_file = Path(rgb_file)
        label_file = Path(label_file)

        rgb = self.load_image(rgb_file) if self.use_image else self.empty_image()
        evt = self.load_event(seq_name, frame_id) if self.use_event else self.empty_event()
        label = self.load_label(label_file)

        return rgb, evt, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_event_cache"] = {}
        return state

    def empty_image(self):
        """Return an image placeholder for event-only ablations."""
        return torch.zeros((self.image_channels, self.image_size[1], self.image_size[0]), dtype=torch.float32)

    def empty_event(self):
        """Return an event placeholder for RGB-only ablations."""
        return torch.zeros(
            (self.event_channels, self.image_size[1], self.image_size[0]),
            dtype=torch.float32,
        )

    def _crop_bottom_image(self, img: Image.Image, image_path: Path) -> Image.Image:
        if self.crop_bottom == 0:
            return img

        width, height = img.size
        if self.crop_bottom >= height:
            raise ValueError(
                f"crop_bottom={self.crop_bottom} is invalid for image '{image_path}' with height={height}"
            )
        return img.crop((0, 0, width, height - self.crop_bottom))

    def load_image(self, image_path):
        if not Path(image_path).exists():
            raise FileNotFoundError(f"DDD17 image file does not exist: {image_path}")
        try:
            with Image.open(image_path) as img:
                if self.force_grayscale:
                    if img.mode != "L":
                        img = img.convert("L")
                    img = self._crop_bottom_image(img, Path(image_path))
                    img = img.resize(self.image_size, Image.BILINEAR)
                    image = TF.to_tensor(img)
                    if self.image_channels == 3:
                        image = image.repeat(3, 1, 1)
                    return image

                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = self._crop_bottom_image(img, Path(image_path))
                img = img.resize(self.image_size, Image.BILINEAR)
                image = TF.to_tensor(img)
                if self.image_channels == 1:
                    image = image.mean(dim=0, keepdim=True)
                return image
        except (IOError, OSError) as err:
            raise RuntimeError(f"Cannot open or process image '{image_path}': {err}") from err

    def load_event(self, seq_name: str, frame_id: int):
        events = None
        if self.use_precomputed_events:
            event_path = self.data_root / seq_name / "event" / f"evt_{frame_id:08d}.npy"
            if event_path.exists():
                events = np.load(event_path)
                if events.size:
                    events = make_relative_events(events[:, :3], events[:, 3])
                else:
                    events = np.empty((0, 4), dtype=np.float32)

        if events is None:
            events = self._load_raw_event_slice(seq_name, frame_id)

        event_frame = build_event_frame(
            events,
            ori_size=self.ori_size,
            frame_type=self.frame_type,
            use_bilinear_voxel=self.use_bilinear_voxel,
        )

        if self.crop_bottom > 0:
            if self.crop_bottom >= event_frame.shape[1]:
                raise ValueError(
                    f"crop_bottom={self.crop_bottom} is invalid for DDD17 event frame "
                    f"seq={seq_name}, frame_id={frame_id}, height={event_frame.shape[1]}"
                )
            event_frame = event_frame[:, :-self.crop_bottom, :]

        event_frame = torch.from_numpy(event_frame)
        out = F.interpolate(
            event_frame[None, ...],
            size=self.image_size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        return out[0]

    def _ensure_event_files(self, seq_name: str):
        cached = self._event_cache.get(seq_name)
        if cached is not None:
            return cached

        seq_dir = self.data_root / seq_name
        index = load_index(seq_dir, self.t_interval)
        t_events, xyp_events = open_event_memmaps(seq_dir)
        cached = {
            "index": index,
            "t_events": t_events,
            "xyp_events": xyp_events,
        }
        self._event_cache[seq_name] = cached
        return cached

    def _load_raw_event_slice(self, seq_name: str, frame_id: int) -> np.ndarray:
        files = self._ensure_event_files(seq_name)
        index = files["index"]
        row_idx = int(frame_id) - 1
        if row_idx < 0 or row_idx >= index.shape[0]:
            raise IndexError(
                f"DDD17 frame_id={frame_id} is outside index range for {seq_name}: "
                f"valid frame ids are 1..{index.shape[0]}"
            )

        _, end_idx, start_idx = index[row_idx]
        start_idx = max(0, int(start_idx))
        event_count = files["t_events"].shape[0]
        if self.event_slice_mode in {"exclusive", "original", "eisnet"}:
            # Original EISNet/ESS slices memmaps as [start_idx:end_idx].
            slice_end = min(int(end_idx), event_count)
        else:
            # Historical CAREFNet behavior included end_idx.
            slice_end = min(int(end_idx) + 1, event_count)

        if slice_end <= start_idx:
            return np.empty((0, 4), dtype=np.float32)

        xyp = np.asarray(files["xyp_events"][start_idx:slice_end], dtype=np.float32)
        timestamps = np.asarray(files["t_events"][start_idx:slice_end, 0], dtype=np.float64)
        return make_relative_events(xyp, timestamps)

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
            raise FileNotFoundError(f"DDD17 label file does not exist: {label_path}")
        with Image.open(label_path) as label:
            label = label.resize(self.image_size, Image.NEAREST)
            label = np.asarray(label, dtype=np.uint8)
            return torch.tensor(label, dtype=torch.long)

    @lru_cache(maxsize=None)
    def _scan_seq(self, seq_dir):
        """Scan one DDD17 sequence once and cache the result."""
        seq_dir = Path(seq_dir)
        masks_dir = seq_dir / "segmentation_masks"
        try:
            entries = sorted(os.scandir(masks_dir), key=lambda e: e.name)
        except FileNotFoundError:
            return [], 0

        samples = []
        missing_images = 0
        for entry in entries:
            if not entry.is_file() or entry.name.startswith(".") or not entry.name.lower().endswith(".png"):
                continue
            frame_id = frame_id_from_mask_name(entry.name)
            if frame_id is None:
                continue
            image_path = find_image_file(seq_dir, frame_id)
            if image_path is None:
                missing_images += 1
                continue
            samples.append((str(image_path), str(seq_dir / "segmentation_masks" / entry.name), seq_dir.name, frame_id))
        return samples, missing_images


if __name__ == "__main__":
    data_check = DDD17Dataset()
    print(len(data_check))
    image, event, labels = data_check[0]
    print(image.shape, image.dtype)
    print(event.shape, event.dtype)
    print(labels.shape, labels.dtype)
