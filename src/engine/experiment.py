"""Experiment directory helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def _resolve_path(path, project_root=None):
    """Resolve relative experiment paths against the project root."""
    path = Path(path).expanduser()
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root).expanduser().resolve() / path


def create_experiment(config, tag='run', resume=None, project_root=None):
    """Create or restore an experiment directory.

    Relative ``config.log_dir`` and ``resume`` paths are resolved against
    ``project_root`` so outputs do not depend on the shell's current directory.
    When resuming, the original ``config.yaml`` is preserved; the current config
    is written to a timestamped ``resume_config_*.yaml`` snapshot instead.
    """
    project_root = Path(project_root).expanduser().resolve() if project_root is not None else None

    if resume is not None:
        logdir = _resolve_path(resume, project_root).resolve()
        if not logdir.exists():
            raise FileNotFoundError(f"Resume experiment directory does not exist: {logdir}")
        if not logdir.is_dir():
            raise NotADirectoryError(f"Resume path is not a directory: {logdir}")
        print("\n==> Restoring experiment from directory:\n" + str(logdir))
    else:
        name = datetime.now().strftime(f"%y-%m-%d--%H-%M-%S_{tag}_{config.train_dataset}_{config.model}")
        logdir = _resolve_path(config.log_dir, project_root).resolve() / name
        print("\n==> Creating new experiment in directory:\n" + str(logdir))
        logdir.mkdir(parents=True, exist_ok=False)

    config_dict = OmegaConf.to_container(config, resolve=True)
    for key, value in config_dict.items():
        print(f"{key}: {value}")

    if resume is None or not (logdir / 'config.yaml').exists():
        config_path = logdir / 'config.yaml'
    else:
        timestamp = datetime.now().strftime("%y-%m-%d--%H-%M-%S")
        config_path = logdir / f'resume_config_{timestamp}.yaml'

    with config_path.open('w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Saved config snapshot: {config_path}")
    return str(logdir)
