"""Training and validation loops."""

from __future__ import annotations

import math

import torch
from tqdm import tqdm

from src.metrics.segmentation import SegMetric


def _dataset_name(config):
    return config.train_dataset if 'train_dataset' in config else 'dataset'


def _move_batch(batch, device):
    return [tensor.to(device, non_blocking=True) for tensor in batch]


def _scheduler_step_per_iteration(config) -> bool:
    scheduler_cfg = config.scheduler if 'scheduler' in config else None
    if scheduler_cfg is None:
        return False
    value = scheduler_cfg.get('step_per_iteration', False)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def train_one_epoch(
    dataloader,
    model,
    criterion,
    optimizer,
    summary,
    config,
    epoch,
    device='cuda:0',
    scheduler=None,
):
    metrics = SegMetric(_dataset_name(config), ignore_index=config.ignore_index if 'ignore_index' in config else 255)
    model.train()
    epoch_loss = 0.0
    iteration = (epoch - 1) * len(dataloader)
    step_scheduler_each_iter = scheduler is not None and _scheduler_step_per_iteration(config)

    for i, batch in enumerate(tqdm(dataloader)):
        image, event, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image, event)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if step_scheduler_each_iter:
            scheduler.step()

        epoch_loss += loss.item()
        metrics.update(logits, labels)

        if i % config.log_interval == 0:
            summary.add_scalar(f'train/{_dataset_name(config)}/loss', float(loss), iteration)
        iteration += 1

    print(f"Epoch {epoch} average loss: {epoch_loss / max(len(dataloader), 1):.4f}")
    display_results(metrics)
    log_results(metrics, _dataset_name(config), summary, 'train', epoch)


def evaluate_one_epoch(dataloader, model, criterion, summary, config, epoch, device='cuda:0'):
    metrics = SegMetric(_dataset_name(config), ignore_index=config.ignore_index if 'ignore_index' in config else 255)
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            image, event, labels = _move_batch(batch, device)
            logits = model(image, event)
            loss = criterion(logits, labels)
            epoch_loss += loss.item()
            metrics.update(logits, labels)

            if i % config.log_interval == 0:
                summary.add_scalar(f'val/{_dataset_name(config)}/loss', float(loss), epoch)

    print(f"Validation average loss: {epoch_loss / max(len(dataloader), 1):.4f}")
    display_results(metrics)
    log_results(metrics, _dataset_name(config), summary, 'val', epoch)
    results = metrics.get_results()
    results['loss'] = epoch_loss / max(len(dataloader), 1)
    return results


def display_results(metrics):
    metrics.print_results()


def log_results(metrics, data_name, summary, split, epoch):
    results = metrics.get_results()
    summary.add_scalar(f'{split}/{data_name}/metrics/miou', round(results['miou'], 4), epoch)
    summary.add_scalar(f'{split}/{data_name}/metrics/pixel_acc', round(results['pixel_acc'], 4), epoch)
    summary.add_scalar(f'{split}/{data_name}/metrics/mean_acc', round(results['mean_acc'], 4), epoch)
    for key, value in results['iou_per_class'].items():
        value = float(value)
        if math.isnan(value):
            continue
        summary.add_scalar(f'{split}/{data_name}/metrics/{key}', round(value, 4), epoch)
    for key, value in results['acc_per_class'].items():
        value = float(value)
        if math.isnan(value):
            continue
        summary.add_scalar(f'{split}/{data_name}/metrics/{key}_acc', round(value, 4), epoch)
