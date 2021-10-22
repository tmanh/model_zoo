import torch
import torch.nn as nn


class SamePaddingConv2d(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, kernel_size // 2, dilation, groups, bias)


class GatingConv2d(SamePaddingConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels+1, kernel_size, stride, kernel_size // 2, dilation, groups, bias)

    def forward(self, x):
        x = super().forward(x)
        return torch.sigmoid(x[:, -1:, :, :]) * x[:, :-1, :, :]
