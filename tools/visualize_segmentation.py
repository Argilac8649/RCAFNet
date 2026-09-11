from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from src.config.load import load_config
from src.datasets.build import build_datasets
from src.datasets.transforms import NormalizedMapDataset
from src.models.build import build_model
from src.utils.device import default_device
from src.utils.visualization import colorize_label


def _as_state_dict(checkpoint: Any):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "state_dict_ema"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _cfg_has(config, key: str) -> bool:
    try:
        return key in config
    except Exception:
        return hasattr(config, key)


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    try:
        return config.get(key, default)
    except Exception:
        return getattr(config, key, default)


def normalize_legacy_config(cfg):
    if not _cfg_has(cfg, "img_size_wh") and _cfg_has(cfg, "img_size"):
        cfg.img_size_wh = list(cfg.img_size)
    if not _cfg_has(cfg, "ori_size_hw") and _cfg_has(cfg, "ori_size"):
        cfg.ori_size_hw = list(cfg.ori_size)
    return cfg


def disable_pretrained(cfg):
    encoder_cfg = _cfg_get(cfg, "encoder", None)
    if encoder_cfg is None:
        return
    for key in ("pretrained", "rgb_pretrained", "event_pretrained"):
        if _cfg_has(encoder_cfg, key):
            encoder_cfg[key] = None


def load_model_checkpoint(model, checkpoint_path: str | Path, device, strict: bool = False):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        best = checkpoint_path / "best.pth"
        latest = checkpoint_path / "latest.pth"
        checkpoint_path = best if best.exists() else latest
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = _as_state_dict(ckpt)
    result = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if hasattr(result, "missing_keys"):
        print(f"Missing keys: {len(result.missing_keys)}")
        print(f"Unexpected keys: {len(result.unexpected_keys)}")


def denormalize_image(image: torch.Tensor, raw_dataset) -> torch.Tensor:
    if not getattr(raw_dataset, "normalize_image", False):
        return image.clamp(0.0, 1.0)
    mean = torch.tensor(raw_dataset.rgb_mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.tensor(raw_dataset.rgb_std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def denormalize_event(event: torch.Tensor, raw_dataset) -> torch.Tensor:
    if not getattr(raw_dataset, "normalize_event", False):
        return event
    mean = torch.tensor(raw_dataset.evt_mean, dtype=event.dtype, device=event.device).view(-1, 1, 1)
    std = torch.tensor(raw_dataset.evt_std, dtype=event.dtype, device=event.device).view(-1, 1, 1)
    return event * std + mean


def image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got {tuple(image.shape)}.")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] > 3:
        image = image[:3]
    return image.permute(1, 2, 0).numpy().clip(0.0, 1.0)


def normalize_map(arr: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(arr, 100.0 - percentile))
    hi = float(np.percentile(arr, percentile))
    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def event_to_heatmap(event: torch.Tensor, raw_dataset, percentile: float) -> np.ndarray:
    event = denormalize_event(event, raw_dataset)
    arr = event.detach().float().abs().mean(dim=0).cpu().numpy()
    return normalize_map(arr, percentile=percentile)


def heatmap_rgb(att_map: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    return plt.get_cmap(cmap_name)(att_map)[..., :3].astype(np.float32)


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def add_panel(ax, title: str, image: np.ndarray):
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_summary_grid(path: Path, panels: list[tuple[str, np.ndarray]], title: str):
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 3.4), dpi=140)
    for ax, (panel_title, image) in zip(np.ravel(axes), panels):
        add_panel(ax, panel_title, image)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def select_indices(dataset_len: int, num_samples: int, random_samples: bool, seed: int, start_index: int, indices: list[int] | None):
    if indices is not None:
        out = [int(v) for v in indices]
        invalid = [v for v in out if v < 0 or v >= dataset_len]
        if invalid:
            raise ValueError(f"Indices outside dataset length {dataset_len}: {invalid}")
        return out
    if random_samples:
        rng = np.random.default_rng(int(seed))
        return [int(v) for v in rng.choice(dataset_len, size=num_samples, replace=False)]
    return list(range(start_index, min(start_index + num_samples, dataset_len)))


def sample_info(raw_val, index: int) -> str:
    items = getattr(raw_val, "data_name", None)
    if items is None or index >= len(items):
        return f"val_index={index}"
    return f"val_index={index}\nsample={items[index]}"


def visualize_one(model, dataset, raw_val, cfg, index: int, sample_dir: Path, device, args):
    image, event, label = dataset[index]
    image_np = image_tensor_to_numpy(denormalize_image(image, raw_val))
    event_heat = heatmap_rgb(event_to_heatmap(event, raw_val, percentile=args.percentile), args.cmap)
    label_np = label.detach().cpu().numpy().astype(np.int64)

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device), event.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)

    dataset_name = str(_cfg_get(cfg, "train_dataset", ""))
    ignore_index = int(_cfg_get(cfg, "ignore_index", 255))
    pred_color = colorize_label(
        pred,
        dataset_name=dataset_name,
        num_classes=int(cfg.num_class),
        ignore_index=ignore_index,
    )
    label_color = colorize_label(
        label_np,
        dataset_name=dataset_name,
        num_classes=int(cfg.num_class),
        ignore_index=ignore_index,
    )
    pred_overlay = np.clip((1.0 - args.alpha) * image_np + args.alpha * pred_color, 0.0, 1.0)
    label_overlay = np.clip((1.0 - args.alpha) * image_np + args.alpha * label_color, 0.0, 1.0)

    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(sample_dir / "rgb.png", image_np)
    save_image(sample_dir / "event_heat.png", event_heat)
    save_image(sample_dir / "prediction.png", pred_color)
    save_image(sample_dir / "label.png", label_color)
    save_image(sample_dir / "prediction_overlay.png", pred_overlay)
    save_image(sample_dir / "label_overlay.png", label_overlay)
    save_summary_grid(
        sample_dir / "summary.png",
        [
            ("RGB", image_np),
            ("Event |mean|", event_heat),
            ("Prediction", pred_color),
            ("Label", label_color),
            ("Pred overlay", pred_overlay),
        ],
        title=f"{str(cfg.model)} - val index {index}",
    )
    (sample_dir / "summary.txt").write_text(sample_info(raw_val, index) + "\n", encoding="utf-8")


def resolve_run_paths(args):
    if args.run_dir:
        run_dir = Path(args.run_dir)
        config = Path(args.config) if args.config else run_dir / "config.yaml"
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pth"
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "segmentation_vis_random10_best"
        return config, checkpoint, output_dir
    if not args.config:
        raise ValueError("Pass either --run_dir or --config.")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "segmentation_vis"
    return Path(args.config), Path(args.checkpoint), output_dir


def main(args):
    config_path, checkpoint_path, output_dir = resolve_run_paths(args)
    device = args.device or default_device()
    cfg = normalize_legacy_config(load_config(config_path))
    disable_pretrained(cfg)

    model = build_model(str(cfg.model), cfg, device=device).eval()
    load_model_checkpoint(model, checkpoint_path, device=device, strict=args.strict)

    _, raw_val = build_datasets(str(cfg.train_dataset), cfg)
    dataset = NormalizedMapDataset(raw_val)
    indices = select_indices(
        len(dataset),
        args.num_samples,
        args.random,
        args.seed,
        args.index,
        args.indices,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / "resolved_config.yaml"))
    (output_dir / "selected_indices.txt").write_text(
        "\n".join(str(v) for v in indices) + "\n",
        encoding="utf-8",
    )

    for order, index in enumerate(indices, start=1):
        sample_dir = output_dir / f"sample_{index:06d}"
        print(f"[{order:03d}/{len(indices):03d}] val index {index} -> {sample_dir}")
        visualize_one(model, dataset, raw_val, cfg, index, sample_dir, device, args)

    print(f"Saved segmentation visualization to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize semantic segmentation predictions.")
    parser.add_argument("--run_dir", type=str, default=None, help="Experiment directory containing config.yaml and best.pth.")
    parser.add_argument("--config", type=str, default=None, help="Config path. Defaults to run_dir/config.yaml.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to run_dir/best.pth.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to run_dir/segmentation_vis_random10_best.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--cmap", type=str, default="turbo")
    parser.add_argument("--percentile", type=float, default=98.0)
    main(parser.parse_args())
