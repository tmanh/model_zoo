import torch
import torch.nn as nn
import torch.nn.functional as functional

from .norm import NormBuilder


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, norm_cfg='BN2d', act=nn.ReLU(inplace=True), bias=False, requires_grad=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, padding=int(dilation * (kernel_size - 1) / 2), dilation=dilation, bias=bias)

        self.act = act
        
        if isinstance(norm_cfg, str):  
            norm_cfg = dict(type=norm_cfg, requires_grad=requires_grad)

        self.bn = NormBuilder.build(cfg=norm_cfg, num_features=out_channels) if norm_cfg else None

        for param in self.conv.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GatingConv2d(nn.Module):
    MODES = ['single', 'full']

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, norm_cfg='BN2d', act=nn.ReLU(inplace=True), bias=False, requires_grad=True, mode='full'):
        super().__init__()

        gating_channels = 1 if mode == 'single' else out_channels
        self.conv = nn.Conv2d(in_channels, out_channels + gating_channels, kernel_size, stride=stride, groups=groups, padding=int(dilation * (kernel_size - 1) / 2), dilation=dilation, bias=bias)

        self.act = act
        
        if isinstance(norm_cfg, str):  
            norm_cfg = dict(type=norm_cfg, requires_grad=requires_grad)

        self.bn = NormBuilder.build(cfg=norm_cfg, num_features=out_channels) if norm_cfg else None

        for param in self.conv.parameters():
            param.requires_grad = requires_grad

        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)

        gated = x[:, :self.out_channels, :, :] * torch.sigmoid(x[:, self.out_channels:, :, :])

        if self.bn is not None:
            gated = self.bn(gated)
        if self.act is not None:
            gated = self.act(gated)
        return gated


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features, conv=DynamicConv2d, norm_cfg=None, act=None, requires_grad=True):
        super().__init__()
        self.convA = conv(skip_input, output_features, kernel_size=3, stride=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)
        self.convB = conv(output_features, output_features, kernel_size=3, stride=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)

    def forward(self, x, concat_with):
        up_x = functional.interpolate(x, size=concat_with.shape[-2:], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))



class UpSampleResidual(nn.Sequential):
    def __init__(self, skip_input, output_features, conv=DynamicConv2d, norm_cfg=None, act=None, requires_grad=True):
        super().__init__()
        self.convA = conv(skip_input, output_features, kernel_size=3, stride=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)
        self.convB = conv(output_features, output_features, kernel_size=3, stride=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)

    def forward(self, x, concat_with):
        print(x.shape, concat_with.shape)
        up_x = functional.interpolate(x, size=concat_with.shape[-2:], mode='bilinear', align_corners=True)
        return self.convB(self.convA(up_x + concat_with))


    
class DownSample(nn.Sequential):
    def __init__(self, in_channels, out_channels, conv=DynamicConv2d, norm_cfg=None, act=None, requires_grad=True):
        super().__init__()
        self.convA = conv(in_channels, out_channels, kernel_size=3, stride=2, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)
        self.convB = conv(out_channels, out_channels, kernel_size=3, stride=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)

    def forward(self, x):
        return self.convB(self.convA(x))



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
        if self.bn is not None:
            x = self.bn(x)
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
