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


def enable_visualization(model):
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "set_visualization"):
        raise RuntimeError("The selected model does not expose encoder.set_visualization().")
    encoder.set_visualization(True)


def get_visualization_cache(model):
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "get_visualization_cache"):
        return {}
    return encoder.get_visualization_cache()


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


def tensor_map_to_array(value, out_hw: tuple[int, int]) -> np.ndarray | None:
    if value is None or not isinstance(value, torch.Tensor):
        return None
    value = value.detach().float().cpu()
    if value.ndim == 5:
        value = value[0]
    if value.ndim == 4:
        value = value[:1]
        if value.shape[1] != 1:
            value = value.abs().mean(dim=1, keepdim=True)
    elif value.ndim == 3:
        value = value[:1].unsqueeze(1)
    elif value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    else:
        return None

    if value.shape[-2:] != out_hw:
        value = F.interpolate(value, size=out_hw, mode="bilinear", align_corners=False)
    arr = value[0, 0].numpy()
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_map(arr: np.ndarray | None, percentile: float = 98.0) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(arr, 100.0 - percentile))
    hi = float(np.percentile(arr, percentile))
    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def normalize_signed_map(arr: np.ndarray | None, percentile: float = 98.0) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.percentile(np.abs(arr), percentile))
    if vmax <= 1e-12:
        vmax = float(np.max(np.abs(arr)))
    if vmax <= 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float32)
    arr = np.clip(arr, -vmax, vmax)
    return ((arr / vmax) + 1.0) * 0.5


def tensor_map_to_numpy(value, out_hw: tuple[int, int], percentile: float = 98.0) -> np.ndarray | None:
    return normalize_map(tensor_map_to_array(value, out_hw), percentile=percentile)


def diff_map_to_numpy(cache: dict, after_key: str, before_key: str, out_hw: tuple[int, int], percentile: float = 98.0):
    after = tensor_map_to_array(cache.get(after_key), out_hw)
    before = tensor_map_to_array(cache.get(before_key), out_hw)
    if after is None or before is None:
        return None
    return normalize_signed_map(after - before, percentile=percentile)


def heatmap_rgb(att_map: np.ndarray | None, cmap_name: str = "turbo") -> np.ndarray:
    if att_map is None:
        return np.ones((1, 1, 3), dtype=np.float32)
    return plt.get_cmap(cmap_name)(att_map)[..., :3].astype(np.float32)


def overlay_heatmap(image_np: np.ndarray, att_map: np.ndarray | None, alpha: float, cmap_name: str) -> np.ndarray:
    if att_map is None:
        return np.ones_like(image_np)
    heat = heatmap_rgb(att_map, cmap_name)
    return np.clip((1.0 - alpha) * image_np + alpha * heat, 0.0, 1.0)


def overlay_signed_heatmap(image_np: np.ndarray, signed_map: np.ndarray | None, alpha: float, cmap_name: str) -> np.ndarray:
    if signed_map is None:
        return np.ones_like(image_np)
    heat = heatmap_rgb(signed_map, cmap_name)
    strength = np.clip(np.abs(signed_map - 0.5) * 2.0, 0.0, 1.0)[..., None]
    weight = alpha * strength
    return np.clip((1.0 - weight) * image_np + weight * heat, 0.0, 1.0)


def event_to_heatmap(event: torch.Tensor, raw_dataset, out_hw: tuple[int, int], percentile: float) -> np.ndarray:
    event = denormalize_event(event, raw_dataset)
    event_map = event.abs().mean(dim=0, keepdim=True).unsqueeze(0)
    return tensor_map_to_numpy(event_map, out_hw, percentile=percentile)


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(image, 0.0, 1.0))


def add_panel(ax, title: str, image: np.ndarray):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def cache_overlay(cache: dict, key: str, image_np: np.ndarray, args):
    att_map = tensor_map_to_numpy(cache.get(key), image_np.shape[:2], percentile=args.percentile)
    return overlay_heatmap(image_np, att_map, alpha=args.alpha, cmap_name=args.cmap)


def cache_heat(cache: dict, key: str, image_np: np.ndarray, args):
    att_map = tensor_map_to_numpy(cache.get(key), image_np.shape[:2], percentile=args.percentile)
    return heatmap_rgb(att_map, args.cmap)


def cache_diff(cache: dict, after_key: str, before_key: str, image_np: np.ndarray, args, overlay: bool = False):
    signed_map = diff_map_to_numpy(cache, after_key, before_key, image_np.shape[:2], percentile=args.percentile)
    if overlay:
        return overlay_signed_heatmap(image_np, signed_map, alpha=args.alpha, cmap_name=args.diff_cmap)
    return heatmap_rgb(signed_map, args.diff_cmap)


def make_stage_panels(stage: dict, image_np: np.ndarray, event_heat: np.ndarray, pred_color: np.ndarray, label_color: np.ndarray, args):
    refiner = stage.get("stage_refiner", {})
    caffm = stage.get("caffm", {})
    panels = [
        ("00_rgb", "RGB", image_np),
        ("01_event_mean", "Event |mean|", heatmap_rgb(event_heat, args.cmap)),
        ("02_prediction", "Prediction", pred_color),
        ("03_label", "Label", label_color),
    ]

    refiner_specs = [
        ("04_refiner_rgb_before", "Refiner RGB before", "rgb_before"),
        ("05_refiner_rgb_after", "Refiner RGB after", "rgb_after"),
        ("06_refiner_event_before", "Refiner Event before", "event_before"),
        ("07_refiner_event_after", "Refiner Event after", "event_after"),
        ("08_refiner_rgb_gate", "Refiner RGB gate calib", "rgb_gate_calibrated"),
        ("09_refiner_event_calib", "Refiner Event calib", "event_to_rgb_calibrated"),
        ("10_refiner_event_to_rgb_update", "Refiner E->RGB update", "event_to_rgb_update"),
        ("11_refiner_quality_event_to_rgb", "Refiner quality E->RGB", "quality_event_to_rgb"),
        ("12_refiner_spatial_event_to_rgb", "Refiner spatial E->RGB", "spatial_event_to_rgb"),
        ("13_refiner_channel_event_to_rgb", "Refiner channel E->RGB", "channel_event_to_rgb_mean"),
    ]
    for stem, title, key in refiner_specs:
        if key in refiner:
            panels.append((stem, title, cache_overlay(refiner, key, image_np, args)))

    caffm_specs = [
        ("14_caffm_rgb_before", "CAFFM RGB before", "rgb_before"),
        ("15_caffm_event_before", "CAFFM Event before", "event_before"),
        ("16_caffm_rgb_after_cross", "CAFFM RGB after cross", "rgb_after_cross"),
        ("17_caffm_event_after_cross", "CAFFM Event after cross", "event_after_cross"),
        ("18_caffm_fused_update", "CAFFM fused update", "fused_update"),
        ("19_caffm_output", "CAFFM output", "output"),
    ]
    for stem, title, key in caffm_specs:
        if key in caffm:
            panels.append((stem, title, cache_overlay(caffm, key, image_np, args)))

    diff_specs = [
        ("20_refiner_rgb_delta", "Delta Refiner RGB", refiner, "rgb_after", "rgb_before"),
        ("21_refiner_event_delta", "Delta Refiner Event", refiner, "event_after", "event_before"),
        ("22_caffm_rgb_cross_delta", "Delta CAFFM RGB cross", caffm, "rgb_after_cross", "rgb_before"),
        ("23_caffm_event_cross_delta", "Delta CAFFM Event cross", caffm, "event_after_cross", "event_before"),
        ("24_caffm_output_delta", "Delta CAFFM output", caffm, "output", "rgb_before"),
    ]
    for stem, title, cache, after_key, before_key in diff_specs:
        if after_key in cache and before_key in cache:
            panels.append((stem, title, cache_diff(cache, after_key, before_key, image_np, args)))
    return panels


def save_stage_outputs(sample_dir: Path, stage: dict, image_np: np.ndarray, event_heat: np.ndarray, pred_color: np.ndarray, label_color: np.ndarray, args):
    stage_id = int(stage.get("stage", 0))
    panels = make_stage_panels(stage, image_np, event_heat, pred_color, label_color, args)

    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.4 * rows), dpi=140)
    for ax, (_, title, panel) in zip(np.ravel(axes), panels):
        add_panel(ax, title, panel)
    for ax in np.ravel(axes)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"RCAFNet attention visualization - stage {stage_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(sample_dir / f"stage_{stage_id}.png")
    plt.close(fig)

    stage_dir = sample_dir / f"stage_{stage_id}"
    panels_dir = stage_dir / "panels"
    for stem, _, panel in panels:
        save_image(panels_dir / f"{stem}.png", panel)

    refiner = stage.get("stage_refiner", {})
    caffm = stage.get("caffm", {})
    heat_specs = [
        ("refiner_rgb_before", refiner, "rgb_before"),
        ("refiner_rgb_after", refiner, "rgb_after"),
        ("refiner_event_before", refiner, "event_before"),
        ("refiner_event_after", refiner, "event_after"),
        ("refiner_rgb_gate_calibrated", refiner, "rgb_gate_calibrated"),
        ("refiner_event_to_rgb_calibrated", refiner, "event_to_rgb_calibrated"),
        ("refiner_event_to_rgb_update", refiner, "event_to_rgb_update"),
        ("refiner_quality_event_to_rgb", refiner, "quality_event_to_rgb"),
        ("refiner_spatial_event_to_rgb", refiner, "spatial_event_to_rgb"),
        ("caffm_rgb_before", caffm, "rgb_before"),
        ("caffm_event_before", caffm, "event_before"),
        ("caffm_rgb_after_cross", caffm, "rgb_after_cross"),
        ("caffm_event_after_cross", caffm, "event_after_cross"),
        ("caffm_fused_update", caffm, "fused_update"),
        ("caffm_output", caffm, "output"),
    ]
    for stem, cache, key in heat_specs:
        if key not in cache:
            continue
        heat = cache_heat(cache, key, image_np, args)
        save_image(stage_dir / "heatmaps" / f"{stem}.png", heat)
        save_image(stage_dir / "heatmaps" / f"{stem}_overlay.png", cache_overlay(cache, key, image_np, args))

    diff_specs = [
        ("refiner_rgb_after_minus_before", refiner, "rgb_after", "rgb_before"),
        ("refiner_event_after_minus_before", refiner, "event_after", "event_before"),
        ("caffm_rgb_after_cross_minus_before", caffm, "rgb_after_cross", "rgb_before"),
        ("caffm_event_after_cross_minus_before", caffm, "event_after_cross", "event_before"),
        ("caffm_output_minus_rgb_before", caffm, "output", "rgb_before"),
    ]
    for stem, cache, after_key, before_key in diff_specs:
        if after_key not in cache or before_key not in cache:
            continue
        save_image(stage_dir / "diffs" / f"{stem}.png", cache_diff(cache, after_key, before_key, image_np, args))
        save_image(stage_dir / "diffs" / f"{stem}_overlay.png", cache_diff(cache, after_key, before_key, image_np, args, overlay=True))


def write_summary(path: Path, cache: dict, sample_info: str):
    lines = [sample_info, ""]
    for stage in cache.get("stages", []):
        lines.append(f"stage {stage.get('stage')}, feature_shape={stage.get('feature_shape')}")
        refiner = stage.get("stage_refiner", {})
        caffm = stage.get("caffm", {})
        if refiner:
            lines.append(f"  Stage refiner module={refiner.get('module')}")
            if "event_to_rgb_gamma" in refiner:
                lines.append(
                    "  Stage refiner event_to_rgb_gamma="
                    f"{_tensor_brief(refiner['event_to_rgb_gamma'])}"
                )
            if "rgb_to_event_gamma" in refiner:
                lines.append(
                    "  Stage refiner rgb_to_event_gamma="
                    f"{_tensor_brief(refiner['rgb_to_event_gamma'])}"
                )
            if "gamma_event" in refiner:
                lines.append(
                    "  Stage refiner gamma_event="
                    f"{_tensor_brief(refiner['gamma_event'])}"
                )
            if "gamma_rgb" in refiner:
                lines.append(
                    "  Stage refiner gamma_rgb="
                    f"{_tensor_brief(refiner['gamma_rgb'])}"
                )
            if "gamma_event_to_rgb" in refiner:
                lines.append(
                    "  Stage refiner gamma_event_to_rgb="
                    f"{_tensor_brief(refiner['gamma_event_to_rgb'])}"
                )
            if "gamma_rgb_to_event" in refiner:
                lines.append(
                    "  Stage refiner gamma_rgb_to_event="
                    f"{_tensor_brief(refiner['gamma_rgb_to_event'])}"
                )
        if caffm:
            lines.append(f"  CAFFM attention_type={caffm.get('attention_type')}, fusion_type={caffm.get('fusion_type')}")
            if "cross_gamma" in caffm:
                lines.append(f"  CAFFM cross_gamma={_tensor_brief(caffm['cross_gamma'])}")
            if "fusion_gamma" in caffm:
                lines.append(f"  CAFFM fusion_gamma={_tensor_brief(caffm['fusion_gamma'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tensor_brief(value):
    if not isinstance(value, torch.Tensor):
        return str(value)
    arr = value.detach().float().cpu().flatten().numpy()
    shown = ", ".join(f"{x:.4f}" for x in arr[:8])
    return f"[{shown}{', ...' if arr.size > 8 else ''}]"


def select_indices(dataset_len: int, num_samples: int, random_samples: bool, seed: int, start_index: int):
    if random_samples:
        rng = np.random.default_rng(int(seed))
        return [int(v) for v in rng.choice(dataset_len, size=num_samples, replace=False)]
    return list(range(start_index, min(start_index + num_samples, dataset_len)))


def sample_info(raw_val, index: int) -> str:
    items = getattr(raw_val, "data_name", None)
    if items is None or index >= len(items):
        return f"val_index={index}"
    item = items[index]
    return f"val_index={index}\nsample={item}"


def visualize_one(model, dataset, raw_val, cfg, index: int, sample_dir: Path, device, args):
    image, event, label = dataset[index]
    out_hw = tuple(image.shape[-2:])
    image_np = image_tensor_to_numpy(denormalize_image(image, raw_val))
    event_heat = event_to_heatmap(event, raw_val, out_hw, percentile=args.percentile)

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device), event.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)

    cache = get_visualization_cache(model)
    stages = cache.get("stages", [])
    if not stages:
        raise RuntimeError("No visualization cache was collected from the model.")

    label_np = label.detach().cpu().numpy().astype(np.int64)
    dataset_name = str(getattr(cfg, "train_dataset", ""))
    ignore_index = int(getattr(cfg, "ignore_index", 255))
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

    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(sample_dir / "rgb.png", image_np)
    save_image(sample_dir / "event_heat.png", heatmap_rgb(event_heat, args.cmap))
    save_image(sample_dir / "prediction.png", pred_color)
    save_image(sample_dir / "label.png", label_color)
    save_image(sample_dir / "prediction_overlay.png", np.clip(0.55 * image_np + 0.45 * pred_color, 0.0, 1.0))
    save_image(sample_dir / "label_overlay.png", np.clip(0.55 * image_np + 0.45 * label_color, 0.0, 1.0))
    write_summary(sample_dir / "summary.txt", cache, sample_info(raw_val, index))

    selected = set(args.stages) if args.stages else None
    for stage in stages:
        stage_id = int(stage.get("stage", 0))
        if selected is not None and stage_id not in selected:
            continue
        save_stage_outputs(sample_dir, stage, image_np, event_heat, pred_color, label_color, args)


def resolve_run_paths(args):
    if args.run_dir:
        run_dir = Path(args.run_dir)
        config = Path(args.config) if args.config else run_dir / "config.yaml"
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pth"
        output_dir = Path(args.output_dir) if args.output_dir else run_dir / "attention_vis_20"
        return config, checkpoint, output_dir
    if not args.config:
        raise ValueError("Pass either --run_dir or --config.")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs" / "attention_vis"
    return Path(args.config), Path(args.checkpoint), output_dir


def main(args):
    config_path, checkpoint_path, output_dir = resolve_run_paths(args)
    device = args.device or default_device()
    cfg = load_config(config_path)

    model = build_model(str(cfg.model), cfg, device=device).eval()
    load_model_checkpoint(model, checkpoint_path, device=device, strict=args.strict)
    enable_visualization(model)

    _, raw_val = build_datasets(str(cfg.train_dataset), cfg)
    dataset = NormalizedMapDataset(raw_val)
    indices = select_indices(len(dataset), args.num_samples, args.random, args.seed, args.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_indices.txt").write_text(
        "\n".join(str(v) for v in indices) + "\n",
        encoding="utf-8",
    )

    for order, index in enumerate(indices, start=1):
        sample_dir = output_dir / f"sample_{index:06d}"
        print(f"[{order:03d}/{len(indices):03d}] val index {index} -> {sample_dir}")
        visualize_one(model, dataset, raw_val, cfg, index, sample_dir, device, args)

    print(f"Saved attention visualization to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RCAFNet recalibration/fusion attention caches.")
    parser.add_argument("--run_dir", type=str, default=None, help="Experiment directory containing config.yaml and best.pth.")
    parser.add_argument("--config", type=str, default=None, help="Config path. Defaults to run_dir/config.yaml.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to run_dir/best.pth.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to run_dir/attention_vis_20.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", type=int, nargs="*", default=None, help="Stages to save, e.g. 1 2 3 4.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--cmap", type=str, default="turbo")
    parser.add_argument("--diff_cmap", type=str, default="coolwarm")
    parser.add_argument("--percentile", type=float, default=98.0)
    main(parser.parse_args())

