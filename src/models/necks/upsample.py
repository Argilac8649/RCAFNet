import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, orisize):
        super(Upsample, self).__init__()
        self.orisize = tuple(orisize[::-1])

    def forward(self, x):
        data = F.interpolate(x, size=self.orisize, mode='bilinear', align_corners=False)
        return data