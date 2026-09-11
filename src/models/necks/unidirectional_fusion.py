import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_


def _maybe_to_cpu(x: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    return x.cpu() if to_cpu else x


def _debug_feature_map(x: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    """将特征张量转换为空间响应图。

    这里使用通道维绝对值均值作为可视化响应，避免依赖具体模块结构。
    默认保留在原设备上，仅在显式要求时搬到 CPU。
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW feature map, got {tuple(x.shape)}.")
    value = x.detach().abs().mean(dim=1, keepdim=True).float()
    return _maybe_to_cpu(value, to_cpu)


def _debug_token_map(
    x: torch.Tensor,
    height: int,
    width: int,
    to_cpu: bool = False,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B, N, C] token map, got {tuple(x.shape)}.")
    batch, tokens, _ = x.shape
    if tokens != int(height) * int(width):
        raise ValueError(
            f"Expected N=H*W for token debug map, got N={tokens}, "
            f"H={height}, W={width}."
        )
    value = x.detach().abs().mean(dim=-1).reshape(batch, 1, height, width).float()
    return _maybe_to_cpu(value, to_cpu)


def _debug_detach(x: torch.Tensor, to_cpu: bool = False) -> torch.Tensor:
    return _maybe_to_cpu(x.detach().float(), to_cpu)


# Fusion

def _to_2tuple(value):
    if (
        isinstance(value, (tuple, list))
        or (
            value is not None
            and not isinstance(value, (str, bytes))
            and hasattr(value, "__iter__")
        )
    ):
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"Expected a 2-tuple/list value, got {value}.")
        return int(values[0]), int(values[1])
    return int(value), int(value)


def _normalize_attention_type(attention_type):
    attention_type = str(attention_type).strip().lower()
    allowed = {"efficient", "window"}
    if attention_type not in allowed:
        raise ValueError(
            f"Unknown attention_type='{attention_type}'. "
            "Expected 'efficient' or 'window'."
        )
    return attention_type


def _check_token_pair(x1, x2, dim: int, module_name: str):
    if x1.ndim != 3 or x2.ndim != 3:
        raise ValueError(
            f"{module_name} expects [B, N, C] tokens, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape != x2.shape:
        raise ValueError(
            f"{module_name} expects matched token shapes, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape[-1] != int(dim):
        raise ValueError(
            f"{module_name} channel mismatch: got C={x1.shape[-1]}, "
            f"expected {int(dim)}."
        )
    return x1.shape


def _check_feature_pair(x1, x2, channels: int, module_name: str):
    if x1.ndim != 4 or x2.ndim != 4:
        raise ValueError(
            f"{module_name} expects NCHW inputs, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape != x2.shape:
        raise ValueError(
            f"{module_name} expects matched feature shapes, got "
            f"x1={tuple(x1.shape)}, x2={tuple(x2.shape)}."
        )
    if x1.shape[1] != int(channels):
        raise ValueError(
            f"{module_name} channel mismatch: got C={x1.shape[1]}, "
            f"expected {int(channels)}."
        )
    return x1.shape