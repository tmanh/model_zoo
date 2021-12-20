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


class SamePaddingConv2dBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=kernel_size // 2, bias=False), act)

    def forward(self, x):
        return self.layers(x)


class SamePaddingNormConv2dBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=int(dilation * (kernel_size - 1) / 2), dilation=dilation, bias=False),
                                    nn.BatchNorm2d(out_channel), act)

    def forward(self, x):
        return self.layers(x)


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, batch_norm=True, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=int(dilation * (kernel_size - 1) / 2), dilation=dilation, bias=False)
        self.act = act
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act=None, batch_norm=False, p_dropout=0.0):
        super().__init__()

        # deconvolution
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=round(kernel_size / 2 - 1))
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None

        self.dropout = nn.Dropout2d(p=p_dropout) if p_dropout > 0 else None

        # activation
        self.act = act

    def forward(self, x):
        x = self.deconv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DeconvGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, stride, channel_wise=True, num_groups=32, group_channels=8, act=None):
        super().__init__()

        # deconvolution
        self.deconv = nn.ConvTranspose2d(in_channels, out_channel, kernel_size, stride=stride, padding=round(kernel_size / 2 - 1))

        if channel_wise:
            num_groups = max(1, int(out_channel / group_channels))
        else:
            num_groups = min(num_groups, out_channel)

        # group norm
        num_channels = int(out_channel // num_groups)
        self.gn = torch.nn.GroupNorm(num_groups, num_channels, affine=channel_wise)

        # activation
        self.act = act

    def forward(self, x):
        x = self.deconv(x)
        
        # group normalization
        x = self.gn(x)

        if self.act is not None:
            x = self.act(x)
        return x
