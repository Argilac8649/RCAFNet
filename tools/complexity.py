from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.config.load import load_config
from src.models.backbones.mit import frame_type_to_event_in_chans
from src.models.build import build_model
from src.utils.device import default_device


def _format_count(value: float) -> str:
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3f}G"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def _shape_of(output):
    if isinstance(output, torch.Tensor):
        return tuple(int(v) for v in output.shape)
    if isinstance(output, (list, tuple)):
        return [_shape_of(v) for v in output]
    if isinstance(output, dict):
        return {key: _shape_of(value) for key, value in output.items()}
    return None


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _disable_pretrained(config):
    encoder_cfg = _cfg_get(config, "encoder", None)
    if encoder_cfg is None:
        return
    for key in ("pretrained", "rgb_pretrained", "event_pretrained"):
        try:
            if key in encoder_cfg:
                encoder_cfg[key] = None
        except Exception:
            if hasattr(encoder_cfg, key):
                setattr(encoder_cfg, key, None)


def _input_hw(config, args):
    if args.input_size is not None:
        height, width = args.input_size
        return int(height), int(width)
    # RCAFNet configs use img_size_wh as [width, height].
    img_size = _cfg_get(config, "img_size_wh", None)
    if img_size is None:
        raise KeyError("Config is missing required key 'img_size_wh'.")
    return int(img_size[1]), int(img_size[0])


def _image_channels(config) -> int:
    encoder_cfg = _cfg_get(config, "encoder", None)
    value = _cfg_get(encoder_cfg, "rgb_in_chans", _cfg_get(config, "image_channels", 3))
    return int(value)


def _event_channels(config) -> int:
    encoder_cfg = _cfg_get(config, "encoder", None)
    value = _cfg_get(encoder_cfg, "event_in_chans", None)
    if value is not None:
        return int(value)
    return int(frame_type_to_event_in_chans(config.frame_type))


def _module_param_count(module: nn.Module, recurse: bool = False) -> int:
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_channels, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    macs_per_output = kernel_h * kernel_w * (module.in_channels // module.groups)
    return int(batch * out_channels * out_h * out_w * macs_per_output)


def linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features)


def batchnorm_macs(output: torch.Tensor) -> int:
    return int(output.numel() * 2)


def layernorm_macs(output: torch.Tensor) -> int:
    return int(output.numel() * 5)


def token_efficient_cross_attention_macs(module: nn.Module, inputs) -> int:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return 0
    x = inputs[0]
    if x.ndim != 3:
        return 0
    batch, tokens, channels = x.shape
    heads = int(getattr(module, "num_heads", 1))
    if heads <= 0:
        return 0
    head_dim = channels // heads
    # Two directions, each with K^T@V and Q@context.
    return int(4 * batch * heads * tokens * head_dim * head_dim)


def token_window_cross_attention_macs(module: nn.Module, inputs) -> int:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return 0
    x = inputs[0]
    if x.ndim != 3:
        return 0
    height = inputs[2] if len(inputs) > 2 else None
    width = inputs[3] if len(inputs) > 3 else None
    if height is None or width is None:
        last_hw = getattr(module, "_last_hw", None)
        if last_hw is not None:
            height, width = last_hw
    if height is None or width is None:
        return 0
    batch, _, channels = x.shape
    heads = int(getattr(module, "num_heads", 1))
    window_h, window_w = getattr(module, "window_size", (1, 1))
    head_dim = channels // heads
    num_windows_h = (int(height) + window_h - 1) // window_h
    num_windows_w = (int(width) + window_w - 1) // window_w
    window_tokens = window_h * window_w
    num_windows = batch * num_windows_h * num_windows_w
    return int(4 * num_windows * heads * window_tokens * window_tokens * head_dim)


def nchw_efficient_cross_attention_macs(module: nn.Module, inputs) -> int:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return 0
    x = inputs[0]
    if x.ndim != 4:
        return 0
    batch, _, height, width = x.shape
    tokens = height * width
    heads = int(getattr(module, "head_count", 1))
    key_channels = int(getattr(module, "key_channels", 0))
    value_channels = int(getattr(module, "value_channels", 0))
    if heads <= 0 or key_channels <= 0 or value_channels <= 0:
        return 0
    key_per_head = key_channels // heads
    value_per_head = value_channels // heads
    # Per head: K@V^T and context^T@Q.
    return int(2 * batch * heads * tokens * key_per_head * value_per_head)


class ComplexityProfiler:
    def __init__(self, include_norm: bool = False):
        self.include_norm = bool(include_norm)
        self.records = []
        self.by_type = defaultdict(int)
        self.handles = []

    def _record(self, name: str, module: nn.Module, macs: int, output):
        if macs <= 0:
            return
        class_name = module.__class__.__name__
        self.records.append({
            "name": name,
            "type": class_name,
            "macs": int(macs),
            "params": int(_module_param_count(module, recurse=False)),
            "output_shape": _shape_of(output),
        })
        self.by_type[class_name] += int(macs)

    def _hook(self, name: str, module: nn.Module):
        def fn(mod, inputs, output):
            macs = 0
            class_name = mod.__class__.__name__
            if isinstance(mod, nn.Conv2d) and isinstance(output, torch.Tensor):
                macs = conv2d_macs(mod, output)
            elif isinstance(mod, nn.Linear) and isinstance(output, torch.Tensor):
                macs = linear_macs(mod, output)
            elif self.include_norm and isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)) and isinstance(output, torch.Tensor):
                macs = batchnorm_macs(output)
            elif self.include_norm and isinstance(mod, nn.LayerNorm) and isinstance(output, torch.Tensor):
                macs = layernorm_macs(output)
            elif class_name == "EfficientTokenCrossAttention":
                macs = token_efficient_cross_attention_macs(mod, inputs)
            elif class_name == "WindowCrossAttention":
                macs = token_window_cross_attention_macs(mod, inputs)
            elif class_name == "EfficientCrossAttention":
                macs = nchw_efficient_cross_attention_macs(mod, inputs)
            self._record(name, mod, macs, output)

        return fn

    def profile(self, model: nn.Module, inputs):
        for name, module in model.named_modules():
            if not name:
                continue
            class_name = module.__class__.__name__
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(self._hook(name, module)))
            elif self.include_norm and isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
                self.handles.append(module.register_forward_hook(self._hook(name, module)))
            elif class_name in {
                "EfficientTokenCrossAttention",
                "WindowCrossAttention",
                "EfficientCrossAttention",
            }:
                self.handles.append(module.register_forward_hook(self._hook(name, module)))

        try:
            with torch.no_grad():
                output = model(*inputs)
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

        return sum(record["macs"] for record in self.records), output


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_latency(model: nn.Module, inputs, device: str, warmup: int, repeats: int):
    if repeats <= 0:
        return None
    with torch.no_grad():
        for _ in range(max(int(warmup), 0)):
            _ = model(*inputs)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(int(repeats)):
            _ = model(*inputs)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / float(repeats)


def print_top_records(records, limit: int):
    if limit <= 0:
        return
    print(f"\nTop {limit} modules by MACs:")
    for record in sorted(records, key=lambda item: item["macs"], reverse=True)[:limit]:
        print(
            f"  {_format_count(record['macs']).rjust(9)} MACs  "
            f"{_format_count(record['params']).rjust(8)} params  "
            f"{record['type']:<24} {record['name']}"
        )


def main(args):
    device = args.device or default_device()
    config = load_config(args.config)
    if not args.load_pretrained:
        _disable_pretrained(config)

    height, width = _input_hw(config, args)
    if args.input_size is not None:
        config.img_size_wh = [width, height]
    batch_size = int(args.batch_size)
    image_ch = _image_channels(config)
    event_ch = _event_channels(config)

    model = build_model(str(config.model), config, device=device).eval()
    image = torch.randn(batch_size, image_ch, height, width, device=device)
    event = torch.randn(batch_size, event_ch, height, width, device=device)

    profiler = ComplexityProfiler(include_norm=args.include_norm)
    macs, output = profiler.profile(model, (image, event))
    params, trainable_params = count_parameters(model)
    latency_ms = measure_latency(model, (image, event), device, args.warmup, args.repeats)
    flops = macs * 2

    result = {
        "model": str(config.model),
        "config": str(args.config),
        "batch_size": batch_size,
        "input_shape": {
            "image": [batch_size, image_ch, height, width],
            "event": [batch_size, event_ch, height, width],
        },
        "output_shape": _shape_of(output),
        "params": int(params),
        "trainable_params": int(trainable_params),
        "macs": int(macs),
        "flops": int(flops),
        "latency_ms": latency_ms,
        "by_type_macs": dict(sorted(profiler.by_type.items(), key=lambda item: item[1], reverse=True)),
        "note": (
            "MACs include Conv2d/Linear and custom cross-attention matrix "
            "multiplications. Elementwise ops, interpolation and most "
            "activations are not counted."
        ),
    }

    print("\nModel complexity")
    print(f"  model            : {result['model']}")
    print(f"  config           : {result['config']}")
    print(f"  input image      : {tuple(result['input_shape']['image'])}")
    print(f"  input event      : {tuple(result['input_shape']['event'])}")
    print(f"  output           : {result['output_shape']}")
    print(f"  params           : {_format_count(params)} ({params:,})")
    print(f"  trainable params : {_format_count(trainable_params)} ({trainable_params:,})")
    print(f"  MACs             : {_format_count(macs)} ({macs:,})")
    print(f"  FLOPs            : {_format_count(flops)} ({flops:,}, estimated as 2 x MACs)")
    if latency_ms is not None:
        print(f"  latency          : {latency_ms:.3f} ms / forward ({args.repeats} repeats)")

    print("\nMACs by module type:")
    for module_type, value in result["by_type_macs"].items():
        print(f"  {module_type:<28}: {_format_count(value)}")
    print_top_records(profiler.records, args.top)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate RCAFNet model parameters and MACs/FLOPs.")
    parser.add_argument("--config", type=str, required=True, help="Complete experiment config YAML.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--load_pretrained", action="store_true")
    parser.add_argument("--include_norm", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    main(parser.parse_args())

