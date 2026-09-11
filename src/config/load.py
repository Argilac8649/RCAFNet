"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def load_config(config_path: str | os.PathLike):
    """Load one complete experiment YAML config."""
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config: {config_path}")
    return cfg


def save_config(config, save_path: str | os.PathLike):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8') as f:
        yaml.dump(OmegaConf.to_container(config, resolve=True), f, default_flow_style=False, sort_keys=False)
