import torch
import torch.nn as nn

from ..basics.dynamic_conv import DynamicConv2d


class SNetDS2BN_base_8(nn.Module):
    """2D U-Net with group normalization."""
    def __init__(self, in_channels):
        super().__init__()

        self.base_channels = 8
        self.conv11 = DynamicConv2d(in_channels, self.base_channels * 1, dilation=1, act=nn.ReLU(inplace=True))
        self.conv12 = DynamicConv2d(self.base_channels * 1, self.base_channels * 2, dilation=1, act=nn.ReLU(inplace=True))
        self.conv13 = DynamicConv2d(self.base_channels * 2, self.base_channels * 2, dilation=2, act=nn.ReLU(inplace=True))
        self.conv14 = DynamicConv2d(self.base_channels * 2, self.base_channels * 2, dilation=1, act=nn.ReLU(inplace=True))

        self.conv21 = DynamicConv2d(self.base_channels * 2, self.base_channels * 2, dilation=3, act=nn.ReLU(inplace=True))
        self.conv22 = DynamicConv2d(self.base_channels * 2, self.base_channels * 2, dilation=1, act=nn.ReLU(inplace=True))

        self.conv31 = DynamicConv2d(self.base_channels * 2, self.base_channels * 2, dilation=4, act=nn.ReLU(inplace=True))
        self.conv32 = DynamicConv2d(self.base_channels * 2, self.base_channels * 2, dilation=1, act=nn.ReLU(inplace=True))

        self.out_conv = DynamicConv2d(self.base_channels * 6, self.base_channels * 2, dilation=1, batch_norm=False, act=None)

    def forward(self, x):
        x11 = self.conv11(x)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        x21 = self.conv21(x13)
        x22 = self.conv21(x21)

        x31 = self.conv31(x13)
        x32 = self.conv32(x31)

        x_merge = torch.cat([x14, x22, x32], dim=1)
        return self.out_conv(x_merge)
