"""Small registry utilities for model builders."""

from __future__ import annotations

from typing import Callable, Dict

MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    def decorator(fn: Callable):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model_builder(name: str) -> Callable:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ', '.join(sorted(MODEL_REGISTRY)) or '<empty>'
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from exc
