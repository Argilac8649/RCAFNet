"""Checkpoint IO utilities."""

from __future__ import annotations

from pathlib import Path
import warnings

import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_miou):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_miou': best_miou,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    if device is None:
        device = next(model.parameters()).device
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as exc:
            warnings.warn(
                "Could not load scheduler state from checkpoint. "
                "This can happen after changing the LR scheduler; "
                "continuing with a fresh scheduler state. "
                f"Original error: {exc}"
            )
    best_miou = ckpt.get('best_miou', 0.0)
    return ckpt.get('epoch', 0), best_miou
