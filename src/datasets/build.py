"""Dataset and DataLoader builders."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .transforms import AugmentedMapDataset, NormalizedMapDataset
from .carla.dataset import CarlaDataset
from .ddd17.dataset import DDD17Dataset
from .dsec.dataset import DsecDataset


def _cfg(config, *names, default=None):
    for name in names:
        if name in config:
            return config[name]
    return default


def _required_cfg(config, name):
    value = _cfg(config, name, default=None)
    if value is None:
        raise KeyError(f"Config is missing required key '{name}'.")
    return value


def _get_aug_value(config, name, default):
    """Read one augmentation option from the nested augmentation config."""
    if 'augmentation' in config and name in config.augmentation:
        return config.augmentation[name]
    return default


def _cfg_split(config, split, name, default=None):
    return _cfg(config, f'{split}_{name}', name, default=default)


def _image_size_wh(config):
    return _required_cfg(config, 'img_size_wh')


def _ori_size_hw(config, default=None):
    return _cfg(config, 'ori_size_hw', default=default)


def _build_dataloader_kwargs(config, split='train'):
    """Build DataLoader keyword arguments for train/val.

    Split-specific options such as ``val_num_workers`` override the shared
    ``num_workers`` setting. This is useful for datasets backed by memmaps or
    native image decoders where validation can occasionally be more stable with
    fewer worker processes.
    """
    num_workers = int(_cfg_split(config, split, 'num_workers', default=0))
    kwargs = {
        'pin_memory': bool(_cfg_split(config, split, 'pin_memory', default=True)),
        'num_workers': num_workers,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(_cfg_split(config, split, 'persistent_workers', default=True))
        kwargs['prefetch_factor'] = int(_cfg_split(config, split, 'prefetch_factor', default=4))
        kwargs['worker_init_fn'] = _seed_worker
        multiprocessing_context = _cfg_split(config, split, 'multiprocessing_context', default=None)
        if multiprocessing_context:
            kwargs['multiprocessing_context'] = str(multiprocessing_context)
    return kwargs


def _seed_worker(worker_id):
    """Seed Python and NumPy RNGs inside DataLoader workers."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed):
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def build_carla_datasets(config):
    print('\n==> Loading CARLA dataset...')
    data_root = _required_cfg(config, 'data_root')
    modality = _cfg(config, 'modality', default='rgb_event')
    train_data = CarlaDataset(
        data_root,
        config.train_folder,
        _image_size_wh(config),
        config.frame_type,
        modality=modality,
    )
    val_data = CarlaDataset(
        data_root,
        config.val_folder,
        _image_size_wh(config),
        config.frame_type,
        modality=modality,
    )
    return train_data, val_data


def build_dsec_datasets(config):
    print('\n==> Loading DSEC dataset...')
    data_root = _required_cfg(config, 'data_root')
    crop_bottom = int(_cfg(config, 'crop_bottom', default=40))
    use_rectify = bool(_cfg(config, 'use_rectify', default=True))
    modality = _cfg(config, 'modality', default='rgb_event')
    normalize_image = bool(_cfg(config, 'normalize_image', default=True))
    normalize_event = bool(_cfg(config, 'normalize_event', default=True))
    use_precomputed_events = bool(_cfg(config, 'use_precomputed_events', default=False))
    precomputed_event_root = _cfg(config, 'precomputed_event_root', default=None)
    train_data = DsecDataset(
        data_root,
        _image_size_wh(config),
        config.frame_type,
        'train',
        crop_bottom=crop_bottom,
        use_rectify=use_rectify,
        modality=modality,
        normalize_image=normalize_image,
        normalize_event=normalize_event,
        use_precomputed_events=use_precomputed_events,
        precomputed_event_root=precomputed_event_root,
    )
    val_data = DsecDataset(
        data_root,
        _image_size_wh(config),
        config.frame_type,
        'val',
        crop_bottom=crop_bottom,
        use_rectify=use_rectify,
        modality=modality,
        normalize_image=normalize_image,
        normalize_event=normalize_event,
        use_precomputed_events=use_precomputed_events,
        precomputed_event_root=precomputed_event_root,
    )
    return train_data, val_data


def build_ddd17_datasets(config):
    print('\n==> Loading DDD17 dataset...')
    data_root = _required_cfg(config, 'data_root')
    crop_bottom = int(_cfg(config, 'crop_bottom', default=60))
    t_interval = int(_cfg(config, 't_interval', default=50))
    ori_size = _ori_size_hw(config, default=[260, 346])
    train_dirs = _cfg(config, 'train_dirs', default=None)
    val_dirs = _cfg(config, 'val_dirs', default=None)
    test_dirs = _cfg(config, 'test_dirs', default=None)
    modality = _cfg(config, 'modality', default='rgb_event')
    force_grayscale = bool(_cfg(config, 'force_grayscale', default=True))
    image_channels = int(_cfg(config, 'image_channels', default=1))
    use_precomputed_events = bool(_cfg(config, 'use_precomputed_events', default=False))
    use_bilinear_voxel = bool(_cfg(config, 'use_bilinear_voxel', default=False))
    event_slice_mode = _cfg(config, 'event_slice_mode', default='inclusive')
    normalize_image = bool(_cfg(config, 'normalize_image', default=True))
    normalize_event = bool(_cfg(config, 'normalize_event', default=True))

    train_data = DDD17Dataset(
        root_folder=data_root,
        image_size=_image_size_wh(config),
        frame_type=config.frame_type,
        mode='train',
        crop_bottom=crop_bottom,
        t_interval=t_interval,
        ori_size=ori_size,
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        test_dirs=test_dirs,
        modality=modality,
        force_grayscale=force_grayscale,
        image_channels=image_channels,
        use_precomputed_events=use_precomputed_events,
        use_bilinear_voxel=use_bilinear_voxel,
        event_slice_mode=event_slice_mode,
        normalize_image=normalize_image,
        normalize_event=normalize_event,
    )
    val_data = DDD17Dataset(
        root_folder=data_root,
        image_size=_image_size_wh(config),
        frame_type=config.frame_type,
        mode='val',
        crop_bottom=crop_bottom,
        t_interval=t_interval,
        ori_size=ori_size,
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        test_dirs=test_dirs,
        modality=modality,
        force_grayscale=force_grayscale,
        image_channels=image_channels,
        use_precomputed_events=use_precomputed_events,
        use_bilinear_voxel=use_bilinear_voxel,
        event_slice_mode=event_slice_mode,
        normalize_image=normalize_image,
        normalize_event=normalize_event,
    )
    return train_data, val_data


def build_datasets(dataset_name, config):
    if dataset_name == 'carla':
        return build_carla_datasets(config)
    if dataset_name == 'dsec':
        return build_dsec_datasets(config)
    if dataset_name == 'ddd17':
        return build_ddd17_datasets(config)
    raise ValueError(f"Unknown dataset option '{dataset_name}'")


def build_trainval_datasets(dataset_name, config):
    train_data, val_data = build_datasets(dataset_name, config)

    train_data = AugmentedMapDataset(
        train_data,
        aug_hflip_prob=_get_aug_value(config, 'hflip_prob', 0.5),
        aug_scale_crop_prob=_get_aug_value(config, 'scale_crop_prob', 0.3),
        scale_range=_get_aug_value(config, 'scale_range', [0.75, 1.5]),
        ignore_index=config.ignore_index if 'ignore_index' in config else 255,
    )
    val_data = NormalizedMapDataset(val_data)
    return train_data, val_data


def build_dataloaders(dataset_name, config):
    train_data, val_data = build_trainval_datasets(dataset_name, config)
    train_loader_kwargs = _build_dataloader_kwargs(config, split='train')
    val_loader_kwargs = _build_dataloader_kwargs(config, split='val')
    seed = _cfg(config, 'seed', default=None)
    epoch_size = _cfg(config, 'epoch_size', default=None)

    train_kwargs = dict(train_loader_kwargs)
    val_kwargs = dict(val_loader_kwargs)
    train_kwargs['generator'] = _make_generator(seed)
    val_kwargs['generator'] = _make_generator(None if seed is None else int(seed) + 1)

    if isinstance(epoch_size, str):
        epoch_size = None if epoch_size.strip().lower() in {'', 'none', 'null', 'full'} else int(epoch_size)

    if epoch_size is None:
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            **train_kwargs,
        )
    else:
        sampler = RandomSampler(
            train_data,
            replacement=True,
            num_samples=int(epoch_size),
            generator=_make_generator(seed),
        )
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=sampler,
            **train_kwargs,
        )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        **val_kwargs,
    )
    return train_loader, val_loader
