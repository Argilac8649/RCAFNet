"""Generic semantic segmentation model wrappers."""

from __future__ import annotations

import torch.nn as nn


class SegNet(nn.Module):
    """Encoder + decode head + final upsample wrapper."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, upsample: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.upsample = upsample

    def forward(self, image, event):
        feature_maps = self.encoder(image, event)
        seg = self.decoder(feature_maps)
        return self.upsample(seg)
