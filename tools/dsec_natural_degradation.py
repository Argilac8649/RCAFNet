"""Audit and evaluate naturally degraded DSEC validation frames.

The workflow is intentionally split into independent stages:

1. ``audit`` reads only RGB/event/label inputs and freezes quality statistics.
2. ``select`` builds degraded subsets and matched controls without predictions.
3. ``evaluate`` reads the frozen manifest and cached model predictions.

This separation prevents model performance from influencing sample selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path[:0] = [str(SOURCE_ROOT), str(PROJECT_ROOT)]

from src.config.load import load_config
from src.datasets.build import build_datasets
from src.utils.visualization import DSEC_CLASS_NAMES


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
QUALITY_FIELDS = (
    "luma_mean",
    "luma_median",
    "luma_p10",
    "luma_p90",
    "luma_p95",
    "luma_p99",
    "dynamic_range",
    "dark_ratio_10",
    "dark_ratio_20",
    "bright_ratio_90",
    "bright_ratio_95",
    "bright_ratio_98",
    "highlight_flat_ratio",
    "perceptual_blur",
    "laplacian_variance",
    "tenengrad",
    "high_frequency_ratio",
    "edge_density",
    "event_mean",
    "event_active_ratio",
)


def sequence_frame_sample(item) -> tuple[str, int, str]:
    path = Path(str(item))
    sequence = path.parents[1].name
    frame = int(path.stem.rsplit("_", 1)[-1])
    return sequence, frame, f"{sequence}/{path.stem}"


def perceptual_blur(gray: np.ndarray, filter_size: int = 9) -> float:
    """Crété-style no-reference blur effect in [0, 1]; higher is blurrier."""
    scores = []
    for axis, kernel in ((0, (1, filter_size)), (1, (filter_size, 1))):
        blurred = cv2.blur(gray, kernel)
        original_variation = np.abs(np.diff(gray, axis=axis))
        blurred_variation = np.abs(np.diff(blurred, axis=axis))
        removable = np.maximum(0.0, original_variation - blurred_variation)
        total = float(original_variation.sum())
        scores.append(0.0 if total <= 1e-12 else (total - float(removable.sum())) / total)
    return float(max(scores))


def high_frequency_ratio(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (160, 110), interpolation=cv2.INTER_AREA)
    spectrum = cv2.dct(small.astype(np.float32))
    energy = spectrum.astype(np.float64) ** 2
    energy[0, 0] = 0.0
    yy, xx = np.indices(energy.shape)
    radius = np.sqrt((yy / energy.shape[0]) ** 2 + (xx / energy.shape[1]) ** 2)
    total = float(energy.sum())
    return 0.0 if total <= 1e-12 else float(energy[radius >= 0.30].sum() / total)


def image_quality(image: torch.Tensor, event: torch.Tensor) -> dict[str, float]:
    rgb = image.detach().float().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
    gray = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)
    gray = cv2.resize(gray, (320, 220), interpolation=cv2.INTER_AREA)
    p10, p50, p90, p95, p99 = np.percentile(gray, (10, 50, 90, 95, 99))
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    bright = gray >= 0.90
    flat = gradient <= 0.015
    event_np = event.detach().float().cpu().numpy()
    return {
        "luma_mean": float(gray.mean()),
        "luma_median": float(p50),
        "luma_p10": float(p10),
        "luma_p90": float(p90),
        "luma_p95": float(p95),
        "luma_p99": float(p99),
        "dynamic_range": float(p90 - p10),
        "dark_ratio_10": float(np.mean(gray <= 0.10)),
        "dark_ratio_20": float(np.mean(gray <= 0.20)),
        "bright_ratio_90": float(np.mean(gray >= 0.90)),
        "bright_ratio_95": float(np.mean(gray >= 0.95)),
        "bright_ratio_98": float(np.mean(gray >= 0.98)),
        "highlight_flat_ratio": float(np.mean(bright & flat)),
        "perceptual_blur": perceptual_blur(gray),
        "laplacian_variance": float(np.var(laplacian)),
        "tenengrad": float(np.mean(gradient * gradient)),
        "high_frequency_ratio": high_frequency_ratio(gray),
        "edge_density": float(np.mean(gradient >= 0.05)),
        "event_mean": float(np.mean(np.abs(event_np))),
        "event_active_ratio": float(np.mean(np.abs(event_np) >= 0.02)),
    }


def label_histogram(label: torch.Tensor, num_classes: int, ignore_index: int) -> np.ndarray:
    array = label.detach().cpu().numpy().astype(np.int64).reshape(-1)
    valid = (array != ignore_index) & (array >= 0) & (array < num_classes)
    counts = np.bincount(array[valid], minlength=num_classes).astype(np.float64)
    return counts / max(1.0, counts.sum())


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def add_composite_scores(rows: list[dict]) -> None:
    by_sequence = defaultdict(list)
    for index, row in enumerate(rows):
        by_sequence[row["sequence"]].append(index)
    for indices in by_sequence.values():
        def metric(name):
            return np.asarray([rows[index][name] for index in indices], dtype=np.float64)

        low_median = 1.0 - rank01(metric("luma_median"))
        low_p10 = 1.0 - rank01(metric("luma_p10"))
        dark20 = rank01(metric("dark_ratio_20"))
        low_p90 = 1.0 - rank01(metric("luma_p90"))
        under = 0.35 * low_median + 0.20 * low_p10 + 0.30 * dark20 + 0.15 * low_p90

        bright95 = rank01(metric("bright_ratio_95"))
        bright98 = rank01(metric("bright_ratio_98"))
        flat_highlight = rank01(metric("highlight_flat_ratio"))
        high_p99 = rank01(metric("luma_p99"))
        over = 0.35 * bright95 + 0.25 * bright98 + 0.25 * flat_highlight + 0.15 * high_p99

        blur = (
            0.55 * rank01(metric("perceptual_blur"))
            + 0.25 * (1.0 - rank01(metric("high_frequency_ratio")))
            + 0.20 * (1.0 - rank01(metric("laplacian_variance")))
        )
        for local, index in enumerate(indices):
            rows[index]["underexposure_score"] = float(under[local])
            rows[index]["overexposure_score"] = float(over[local])
            rows[index]["motion_blur_score"] = float(blur[local])


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(rows: list[dict], score_name: str, output: Path, count: int = 24):
    selected = sorted(rows, key=lambda row: float(row[score_name]), reverse=True)[:count]
    columns, tile_width, tile_height, label_height = 4, 320, 220, 54
    rows_count = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows_count * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 16) if FONT.is_file() else ImageFont.load_default()
    for order, row in enumerate(selected):
        x = (order % columns) * tile_width
        y = (order // columns) * (tile_height + label_height)
        with Image.open(row["source_image"]) as image:
            image = image.convert("RGB")
            image = image.crop((0, 0, image.width, image.height - 40))
            image = image.resize((tile_width, tile_height), Image.Resampling.BILINEAR)
            sheet.paste(image, (x, y))
        label = (
            f"{row['sample_id']}  score={float(row[score_name]):.3f}\n"
            f"Y50={float(row['luma_median']):.3f}  blur={float(row['perceptual_blur']):.3f}  "
            f"clip98={100*float(row['bright_ratio_98']):.2f}%"
        )
        draw.multiline_text((x + 5, y + tile_height + 3), label, font=font, fill=(35, 43, 55), spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def audit(args):
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if str(config.train_dataset).lower() != "dsec":
        raise ValueError("The audit requires a DSEC configuration.")
    _, raw_validation = build_datasets("dsec", config)
    loader = DataLoader(
        raw_validation,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        persistent_workers=int(args.num_workers) > 0,
    )
    items = raw_validation.data_name
    num_classes = int(config.num_class)
    ignore_index = int(config.ignore_index)
    rows = []
    histograms = []
    for index, (image, event, label) in enumerate(tqdm(loader, desc="DSEC quality audit")):
        sequence, frame, sample_id = sequence_frame_sample(items[index])
        row = {
            "index": index,
            "sequence": sequence,
            "frame": frame,
            "sample_id": sample_id,
            "source_image": str(items[index]),
            **image_quality(image[0], event[0]),
        }
        rows.append(row)
        histograms.append(label_histogram(label[0], num_classes, ignore_index))
    add_composite_scores(rows)
    write_rows(output / "quality_scores.csv", rows)
    np.save(output / "label_histograms.npy", np.asarray(histograms, dtype=np.float32))
    for score in ("underexposure_score", "overexposure_score", "motion_blur_score"):
        make_contact_sheet(rows, score, output / f"top_{score}.png")

    summary = {"samples": len(rows), "sequences": {}, "quality_quantiles": {}}
    for sequence in sorted({row["sequence"] for row in rows}):
        summary["sequences"][sequence] = sum(row["sequence"] == sequence for row in rows)
    for field in QUALITY_FIELDS + (
        "underexposure_score",
        "overexposure_score",
        "motion_blur_score",
    ):
        values = np.asarray([float(row[field]) for row in rows])
        summary["quality_quantiles"][field] = {
            str(q): float(np.quantile(values, q)) for q in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
        }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved prediction-independent audit to {output}")


def load_quality_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {"index", "frame"}
    string_fields = {"sequence", "sample_id", "source_image"}
    for row in rows:
        for key, value in list(row.items()):
            if key in integer_fields:
                row[key] = int(value)
            elif key not in string_fields:
                row[key] = float(value)
    return rows


def standardized_features(rows, histograms, indices, condition: str) -> np.ndarray:
    hist = histograms[indices].astype(np.float64)
    event = np.asarray([rows[index]["event_active_ratio"] for index in indices])[:, None]
    luminance = np.asarray([rows[index]["luma_median"] for index in indices])[:, None]
    # Match event activity for blur and semantic distribution for every condition.
    features = [hist * 2.0, event]
    if condition == "motion_blur":
        features.append(luminance)
    matrix = np.concatenate(features, axis=1)
    scale = np.std(matrix, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return matrix / scale


def matched_controls(rows, histograms, degraded: list[int], candidates: list[int], condition: str):
    selected_controls = []
    for sequence in sorted({rows[index]["sequence"] for index in degraded}):
        deg = [index for index in degraded if rows[index]["sequence"] == sequence]
        pool = [index for index in candidates if rows[index]["sequence"] == sequence]
        if len(pool) < len(deg):
            raise RuntimeError(f"Insufficient control candidates for {condition}/{sequence}.")
        combined = deg + pool
        features = standardized_features(rows, histograms, combined, condition)
        deg_features = features[: len(deg)]
        pool_features = features[len(deg):]
        cost = np.mean(np.abs(deg_features[:, None, :] - pool_features[None, :, :]), axis=2)
        row_indices, column_indices = linear_sum_assignment(cost)
        if len(row_indices) != len(deg):
            raise RuntimeError("Control matching did not cover every degraded sample.")
        selected_controls.extend(pool[column] for column in column_indices)
    return selected_controls


def select(args):
    output = Path(args.output_dir).expanduser().resolve()
    rows = load_quality_rows(output / "quality_scores.csv")
    histograms = np.load(output / "label_histograms.npy")
    fraction = float(args.fraction)
    conditions = {
        "underexposure": "underexposure_score",
        "overexposure": "overexposure_score",
        "motion_blur": "motion_blur_score",
    }
    manifest = {
        "selection_uses_predictions": False,
        "fraction_per_sequence": fraction,
        "conditions": {},
    }
    all_indices = set(range(len(rows)))
    for condition, score_name in conditions.items():
        degraded = []
        for sequence in sorted({row["sequence"] for row in rows}):
            sequence_indices = [index for index, row in enumerate(rows) if row["sequence"] == sequence]
            eligible = sequence_indices
            if condition == "motion_blur":
                edge_floor = np.quantile([rows[index]["edge_density"] for index in sequence_indices], 0.20)
                event_floor = np.quantile([rows[index]["event_active_ratio"] for index in sequence_indices], 0.40)
                eligible = [
                    index for index in sequence_indices
                    if rows[index]["edge_density"] >= edge_floor
                    and rows[index]["event_active_ratio"] >= event_floor
                ]
            count = max(1, round(fraction * len(sequence_indices)))
            ranked = sorted(eligible, key=lambda index: rows[index][score_name], reverse=True)
            degraded.extend(ranked[:count])

        degraded_set = set(degraded)
        normal_candidates = []
        for sequence in sorted({row["sequence"] for row in rows}):
            sequence_indices = [index for index, row in enumerate(rows) if row["sequence"] == sequence]
            values = np.asarray([rows[index][score_name] for index in sequence_indices])
            low, high = np.quantile(values, (0.30, 0.70))
            normal_candidates.extend(
                index for index in sequence_indices
                if low <= rows[index][score_name] <= high and index not in degraded_set
            )
        controls = matched_controls(rows, histograms, degraded, normal_candidates, condition)
        manifest["conditions"][condition] = {
            "score": score_name,
            "degraded_indices": sorted(degraded),
            "control_indices": sorted(controls),
            "degraded_samples": [rows[index]["sample_id"] for index in sorted(degraded)],
            "control_samples": [rows[index]["sample_id"] for index in sorted(controls)],
            "count": len(degraded),
        }
    manifest["overlap_counts"] = {
        f"{a}__{b}": len(
            set(manifest["conditions"][a]["degraded_indices"])
            & set(manifest["conditions"][b]["degraded_indices"])
        )
        for i, a in enumerate(conditions)
        for b in list(conditions)[i + 1 :]
    }
    manifest_path = output / "frozen_subsets.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frozen prediction-independent subsets: {manifest_path}")


def confusion_for_subset(predictions, labels, indices, num_classes=11, ignore_index=255):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for start in range(0, len(indices), 32):
        batch_indices = indices[start : start + 32]
        pred = np.asarray(predictions[batch_indices], dtype=np.int64).reshape(-1)
        target = np.asarray(labels[batch_indices], dtype=np.int64).reshape(-1)
        valid = (
            (target != ignore_index)
            & (target >= 0)
            & (target < num_classes)
            & (pred >= 0)
            & (pred < num_classes)
        )
        encoded = num_classes * target[valid] + pred[valid]
        confusion += np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return confusion


def metrics_from_confusion(confusion):
    confusion = confusion.astype(np.float64)
    tp = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - tp
    valid = union > 0
    iou = np.full(len(union), np.nan)
    iou[valid] = tp[valid] / union[valid]
    return {
        "miou": float(np.nanmean(iou)),
        "pixel_accuracy": float(tp.sum() / max(1.0, confusion.sum())),
        "per_class_iou": {
            name: None if not np.isfinite(iou[index]) else float(iou[index])
            for index, name in enumerate(DSEC_CLASS_NAMES)
        },
    }


def load_cache(path: Path, require_labels: bool = True):
    predictions = np.load(path / "predictions.npy", mmap_mode="r")
    label_path = path / "labels.npy"
    labels = np.load(label_path, mmap_mode="r") if label_path.exists() else None
    if require_labels and labels is None:
        raise FileNotFoundError(f"Missing labels: {label_path}")
    sample_ids = json.loads((path / "sample_ids.json").read_text(encoding="utf-8"))
    return predictions, labels, sample_ids


def evaluate(args):
    output = Path(args.output_dir).expanduser().resolve()
    manifest = json.loads((output / "frozen_subsets.json").read_text(encoding="utf-8"))
    image_pred, image_labels, image_ids = load_cache(
        Path(args.image_only_cache).expanduser().resolve(), require_labels=False
    )
    rcaf_pred, rcaf_labels, rcaf_ids = load_cache(Path(args.rcafnet_cache).expanduser().resolve())
    if image_ids != rcaf_ids:
        raise RuntimeError("Image-only and RCAFNet caches are not sample aligned.")
    if image_pred.shape != rcaf_pred.shape:
        raise RuntimeError("Image-only and RCAFNet prediction shapes are not aligned.")
    if image_labels is not None:
        if image_labels.shape != rcaf_labels.shape:
            raise RuntimeError("Image-only and RCAFNet label shapes are not aligned.")
        for start in range(0, len(image_labels), 32):
            if not np.array_equal(image_labels[start : start + 32], rcaf_labels[start : start + 32]):
                raise RuntimeError("Image-only and RCAFNet labels are not aligned.")
    labels = rcaf_labels

    results = {"metric_protocol": "aggregate confusion matrix", "conditions": {}}
    csv_rows = []
    for condition, specification in manifest["conditions"].items():
        condition_result = {}
        for subset_name, indices_key in (("degraded", "degraded_indices"), ("control", "control_indices")):
            indices = specification[indices_key]
            image_metrics = metrics_from_confusion(confusion_for_subset(image_pred, labels, indices))
            rcaf_metrics = metrics_from_confusion(confusion_for_subset(rcaf_pred, labels, indices))
            condition_result[subset_name] = {
                "count": len(indices),
                "image_only": image_metrics,
                "rcafnet": rcaf_metrics,
                "rcafnet_minus_image_only": rcaf_metrics["miou"] - image_metrics["miou"],
            }
        condition_result["complementarity_gain"] = (
            condition_result["degraded"]["rcafnet_minus_image_only"]
            - condition_result["control"]["rcafnet_minus_image_only"]
        )
        results["conditions"][condition] = condition_result
        csv_rows.append(
            {
                "condition": condition,
                "count": condition_result["degraded"]["count"],
                "degraded_image_only_miou": condition_result["degraded"]["image_only"]["miou"],
                "degraded_rcafnet_miou": condition_result["degraded"]["rcafnet"]["miou"],
                "degraded_model_gap": condition_result["degraded"]["rcafnet_minus_image_only"],
                "control_image_only_miou": condition_result["control"]["image_only"]["miou"],
                "control_rcafnet_miou": condition_result["control"]["rcafnet"]["miou"],
                "control_model_gap": condition_result["control"]["rcafnet_minus_image_only"],
                "complementarity_gain": condition_result["complementarity_gain"],
            }
        )
    (output / "natural_degradation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rows(output / "natural_degradation_results.csv", csv_rows)
    print(json.dumps(csv_rows, ensure_ascii=False, indent=2))


def _block_bootstrap_mean(
    values: np.ndarray,
    sample_indices: list[int],
    rows: list[dict],
    random_generator: np.random.Generator,
    block_length: int,
) -> float:
    sampled = []
    for sequence in sorted({rows[index]["sequence"] for index in sample_indices}):
        sequence_indices = sorted(
            (index for index in sample_indices if rows[index]["sequence"] == sequence),
            key=lambda index: rows[index]["frame"],
        )
        sequence_values = values[sequence_indices]
        count = len(sequence_values)
        blocks = []
        while sum(len(block) for block in blocks) < count:
            start = int(random_generator.integers(0, count))
            offsets = (start + np.arange(min(block_length, count))) % count
            blocks.append(sequence_values[offsets])
        sampled.append(np.concatenate(blocks)[:count])
    return float(np.concatenate(sampled).mean())


def bootstrap_interval(
    values: np.ndarray,
    sample_indices: list[int],
    rows: list[dict],
    repetitions: int,
    block_length: int,
    seed: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            _block_bootstrap_mean(values, sample_indices, rows, generator, block_length)
            for _ in range(repetitions)
        ]
    )
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def evaluate_scores(args):
    """Evaluate frozen subsets from existing per-frame prediction scores only."""
    output = Path(args.output_dir).expanduser().resolve()
    quality_rows = load_quality_rows(output / "quality_scores.csv")
    manifest = json.loads((output / "frozen_subsets.json").read_text(encoding="utf-8"))
    with Path(args.scores_csv).expanduser().resolve().open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        score_rows = list(csv.DictReader(handle))
    if len(score_rows) != len(quality_rows):
        raise RuntimeError("Quality audit and prediction-score lengths differ.")
    if any(
        score.get("sample_id", quality["sample_id"]) != quality["sample_id"]
        for score, quality in zip(score_rows, quality_rows)
    ):
        raise RuntimeError("Quality audit and prediction scores are not sample aligned.")

    image_only = np.asarray([float(row["image_only_miou"]) for row in score_rows])
    rcafnet = np.asarray([float(row["rcafnet_miou"]) for row in score_rows])
    paired_difference = rcafnet - image_only
    repetitions = int(args.bootstrap_repetitions)
    block_length = int(args.block_length)
    results = {
        "metric_protocol": "mean of per-frame mIoU values",
        "selection_uses_predictions": False,
        "scores_csv": str(Path(args.scores_csv).expanduser().resolve()),
        "bootstrap": {
            "type": "sequence-stratified circular moving-block bootstrap",
            "repetitions": repetitions,
            "block_length": block_length,
            "seed": int(args.seed),
        },
        "full_validation": {
            "count": len(score_rows),
            "image_only_miou": float(image_only.mean()),
            "rcafnet_miou": float(rcafnet.mean()),
            "model_gap": float(paired_difference.mean()),
        },
        "conditions": {},
    }
    csv_rows = []
    for condition_index, (condition, specification) in enumerate(manifest["conditions"].items()):
        condition_result = {}
        for subset_name, indices_key in (("degraded", "degraded_indices"), ("control", "control_indices")):
            indices = [int(index) for index in specification[indices_key]]
            gap = float(paired_difference[indices].mean())
            per_sequence = {}
            for sequence in sorted({quality_rows[index]["sequence"] for index in indices}):
                sequence_indices = [
                    index for index in indices if quality_rows[index]["sequence"] == sequence
                ]
                per_sequence[sequence] = {
                    "count": len(sequence_indices),
                    "image_only_miou": float(image_only[sequence_indices].mean()),
                    "rcafnet_miou": float(rcafnet[sequence_indices].mean()),
                    "model_gap": float(paired_difference[sequence_indices].mean()),
                }
            condition_result[subset_name] = {
                "count": len(indices),
                "image_only_miou": float(image_only[indices].mean()),
                "rcafnet_miou": float(rcafnet[indices].mean()),
                "model_gap": gap,
                "model_gap_95ci": bootstrap_interval(
                    paired_difference,
                    indices,
                    quality_rows,
                    repetitions,
                    block_length,
                    int(args.seed) + 100 * condition_index + (0 if subset_name == "degraded" else 1),
                ),
                "per_sequence": per_sequence,
            }
        complementarity_gain = (
            condition_result["degraded"]["model_gap"]
            - condition_result["control"]["model_gap"]
        )
        generator = np.random.default_rng(int(args.seed) + 1000 + condition_index)
        complementarity_samples = []
        degraded_indices = specification["degraded_indices"]
        control_indices = specification["control_indices"]
        for _ in range(repetitions):
            degraded_mean = _block_bootstrap_mean(
                paired_difference, degraded_indices, quality_rows, generator, block_length
            )
            control_mean = _block_bootstrap_mean(
                paired_difference, control_indices, quality_rows, generator, block_length
            )
            complementarity_samples.append(degraded_mean - control_mean)
        complementarity_ci = [
            float(value) for value in np.quantile(complementarity_samples, (0.025, 0.975))
        ]
        condition_result["complementarity_gain"] = complementarity_gain
        condition_result["complementarity_gain_95ci"] = complementarity_ci
        results["conditions"][condition] = condition_result
        csv_rows.append(
            {
                "condition": condition,
                "count": condition_result["degraded"]["count"],
                "degraded_image_only_miou": condition_result["degraded"]["image_only_miou"],
                "degraded_rcafnet_miou": condition_result["degraded"]["rcafnet_miou"],
                "degraded_model_gap": condition_result["degraded"]["model_gap"],
                "degraded_gap_ci_low": condition_result["degraded"]["model_gap_95ci"][0],
                "degraded_gap_ci_high": condition_result["degraded"]["model_gap_95ci"][1],
                "control_model_gap": condition_result["control"]["model_gap"],
                "complementarity_gain": complementarity_gain,
                "complementarity_ci_low": complementarity_ci[0],
                "complementarity_ci_high": complementarity_ci[1],
            }
        )
    (output / "natural_degradation_score_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rows(output / "natural_degradation_score_results.csv", csv_rows)
    print(json.dumps(csv_rows, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="DSEC natural degradation subset analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--config", required=True)
    audit_parser.add_argument("--output_dir", required=True)
    audit_parser.add_argument("--num_workers", type=int, default=8)
    audit_parser.set_defaults(function=audit)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--output_dir", required=True)
    select_parser.add_argument("--fraction", type=float, default=0.10)
    select_parser.set_defaults(function=select)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output_dir", required=True)
    evaluate_parser.add_argument("--image_only_cache", required=True)
    evaluate_parser.add_argument("--rcafnet_cache", required=True)
    evaluate_parser.set_defaults(function=evaluate)

    score_parser = subparsers.add_parser("evaluate_scores")
    score_parser.add_argument("--output_dir", required=True)
    score_parser.add_argument("--scores_csv", required=True)
    score_parser.add_argument("--bootstrap_repetitions", type=int, default=5000)
    score_parser.add_argument("--block_length", type=int, default=20)
    score_parser.add_argument("--seed", type=int, default=20260907)
    score_parser.set_defaults(function=evaluate_scores)
    args = parser.parse_args()
    if hasattr(args, "fraction") and not 0.0 < args.fraction < 0.5:
        parser.error("--fraction must be in (0, 0.5).")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
