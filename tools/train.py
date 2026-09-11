from __future__ import annotations

import argparse
import faulthandler
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src.config.load import load_config
from src.datasets.build import build_dataloaders
from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.engine.experiment import create_experiment
from src.engine.trainer import evaluate_one_epoch, train_one_epoch
from src.losses import build_loss
from src.models.build import build_model
from src.utils.device import default_device
from src.utils.seed import seed_everything


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _scheduler_get(config, key, default=None):
    scheduler_cfg = _cfg_get(config, "scheduler", None)
    value = _cfg_get(scheduler_cfg, key, None)
    if value is not None:
        return value
    return _cfg_get(config, key, default)


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def build_scheduler(optimizer, config, steps_per_epoch: int = 1):
    scheduler_type = str(_scheduler_get(config, "type", "warmup_cosine")).lower()
    steps_per_epoch = max(int(steps_per_epoch), 1)

    if scheduler_type in {"poly", "polynomial", "polynomial_lr"}:
        step_per_iteration = _to_bool(_scheduler_get(config, "step_per_iteration", True))
        total_steps = int(config.num_epochs) * (steps_per_epoch if step_per_iteration else 1)
        total_steps = max(total_steps, 1)
        power = float(_scheduler_get(config, "power", 1.0))

        def lr_lambda(step_idx: int):
            progress = min(max(int(step_idx), 0), total_steps) / float(total_steps)
            return (1.0 - progress) ** power

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    if scheduler_type not in {"warmup_cosine", "warmup+cosine", "cosine_warmup"}:
        raise ValueError(
            f"Unsupported scheduler.type='{scheduler_type}'. "
            "Available: warmup_cosine, polynomial."
        )

    num_epochs = int(config.num_epochs)
    warmup_epochs = int(_scheduler_get(config, "warmup_epochs", 5))
    eta_min = float(_scheduler_get(config, "eta_min", _scheduler_get(config, "min_lr", 1e-6)))
    warmup_start_factor = float(_scheduler_get(config, "warmup_start_factor", 0.01))

    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}.")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}.")
    if not (0.0 < warmup_start_factor <= 1.0):
        raise ValueError(
            "warmup_start_factor must be in (0, 1], "
            f"got {warmup_start_factor}."
        )

    if num_epochs > 1:
        warmup_epochs = min(warmup_epochs, num_epochs - 1)
    else:
        warmup_epochs = 0

    base_lr = float(config.learning_rate)
    eta_min_factor = eta_min / base_lr if base_lr > 0 else 0.0
    eta_min_factor = min(max(eta_min_factor, 0.0), 1.0)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)

    def lr_lambda(step_idx: int):
        step_idx = int(step_idx)
        if warmup_epochs > 0 and step_idx < warmup_epochs:
            progress = step_idx / float(warmup_epochs)
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress

        progress = (step_idx - warmup_epochs) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine_factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def _format_metric_value(value):
    value = float(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.4f}"


def format_best_metrics(record):
    results = record["results"]
    iou_per_class = results.get("iou_per_class", {})
    acc_per_class = results.get("acc_per_class", {})
    label_width = max(
        len(name)
        for name in list(iou_per_class.keys()) + list(acc_per_class.keys()) + [
            "mIoU",
            "Pixel Acc",
            "Mean Acc",
            "Val Loss",
        ]
    ) + 2

    lines = [
        "Best Validation Metrics",
        f"Epoch       : {record['epoch']}",
        f"Model       : {record['model']}",
        f"Dataset     : {record['dataset']}",
        "",
        "Per-class IoU",
    ]
    for name, value in iou_per_class.items():
        lines.append(f"{name:<{label_width}}: {_format_metric_value(value)}")

    lines.extend([
        "",
        "Summary",
        f"{'mIoU':<{label_width}}: {_format_metric_value(results['miou'])}",
        f"{'Pixel Acc':<{label_width}}: {_format_metric_value(results['pixel_acc'])}",
        f"{'Mean Acc':<{label_width}}: {_format_metric_value(results['mean_acc'])}",
        f"{'Val Loss':<{label_width}}: {_format_metric_value(results.get('loss', float('nan')))}",
        "",
        "Per-class Acc",
    ])
    for name, value in acc_per_class.items():
        lines.append(f"{name:<{label_width}}: {_format_metric_value(value)}")
    lines.append("")
    return "\n".join(lines)


def save_best_metrics_txt(logdir, record):
    path = Path(logdir) / "best_val_metrics.txt"
    path.write_text(format_best_metrics(record), encoding="utf-8")
    print(f"Saved best validation metrics: {path}")


def resolve_config_path(config_path: str | os.PathLike) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main(args):
    faulthandler.enable()

    config = load_config(resolve_config_path(args.config))
    seed = args.seed if args.seed is not None else _cfg_get(config, "seed", None)
    if seed is not None and str(seed).strip().lower() not in {"", "none", "null", "false"}:
        config.seed = int(seed)
        seed_everything(config.seed, deterministic=args.deterministic)

    device = args.device or default_device()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = not args.deterministic

    model_name = str(config.model)
    dataset_name = str(config.train_dataset)

    logdir = create_experiment(config, args.tag, args.resume, project_root=PROJECT_ROOT)
    summary = SummaryWriter(logdir)

    model = build_model(model_name, config, device=device)
    criterion = build_loss(config)
    print(f"Loss: {criterion.__class__.__name__}")

    train_loader, val_loader = build_dataloaders(dataset_name, config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    scheduler_type = str(_scheduler_get(config, "type", "warmup_cosine")).lower()
    if scheduler_type in {"poly", "polynomial", "polynomial_lr"}:
        print(
            "Scheduler: polynomial "
            f"(base_lr={float(config.learning_rate):.6g}, "
            f"initial_lr={current_lr(optimizer):.6g}, "
            f"power={float(_scheduler_get(config, 'power', 1.0))}, "
            f"step_per_iteration={_to_bool(_scheduler_get(config, 'step_per_iteration', True))})"
        )
    else:
        print(
            "Scheduler: warmup + cosine "
            f"(base_lr={float(config.learning_rate):.6g}, "
            f"initial_lr={current_lr(optimizer):.6g}, "
            f"eta_min={float(_scheduler_get(config, 'eta_min', _scheduler_get(config, 'min_lr', 1e-6))):.6g}, "
            f"warmup_epochs={int(_scheduler_get(config, 'warmup_epochs', 5))})"
        )

    if args.resume:
        epoch, best_miou = load_checkpoint(
            os.path.join(logdir, "latest.pth"),
            model,
            optimizer,
            scheduler,
            device=device,
        )
        epoch += 1
        print(f"Resuming from epoch {epoch}, best mIoU: {best_miou:.4f}")
    else:
        epoch, best_miou = 1, float("-inf")
    best_val_record = None

    while epoch <= config.num_epochs:
        print(f"\n\n=== Beginning epoch {epoch} of {config.num_epochs} ===")
        print(f"Learning rate: {current_lr(optimizer):.6g}")
        summary.add_scalar("train/lr", current_lr(optimizer), epoch)

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            summary,
            config,
            epoch,
            device=device,
            scheduler=scheduler,
        )
        val_results = evaluate_one_epoch(
            val_loader,
            model,
            criterion,
            summary,
            config,
            epoch,
            device=device,
        )
        val_miou = float(val_results["miou"])
        if not _to_bool(_scheduler_get(config, "step_per_iteration", False)):
            scheduler.step()

        if val_miou > best_miou:
            best_miou = val_miou
            best_val_record = {
                "epoch": epoch,
                "model": model_name,
                "dataset": dataset_name,
                "results": val_results,
            }
            save_best_metrics_txt(logdir, best_val_record)
            save_checkpoint(
                os.path.join(logdir, "best.pth"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_miou,
            )
        save_checkpoint(
            os.path.join(logdir, "latest.pth"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_miou,
        )
        epoch += 1

    if best_val_record is not None:
        save_best_metrics_txt(logdir, best_val_record)
    summary.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="train", help="optional tag to identify the run")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rcafnet_dsec.yaml",
        help="path to one complete experiment YAML config",
    )
    parser.add_argument("--resume", type=str, default=None, help="path to an experiment directory to resume")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    main(parser.parse_args())

