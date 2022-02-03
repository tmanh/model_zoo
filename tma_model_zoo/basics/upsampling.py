import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()

        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x, size=None):
        return nn.functional.interpolate(x, size=size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
