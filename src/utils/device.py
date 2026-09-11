"""Device helpers."""

from __future__ import annotations

import torch


def default_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
