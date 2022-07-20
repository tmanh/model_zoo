import torch
import torch.nn as nn
import torch.nn.functional as functional

from .base import *

from ..universal.resnet import Resnet
from ..basics.upsampling import Upscale
from ..basics.dynamic_conv import GatingConv2d, DynamicConv2d, DownSample, UpSample


class DepthEncoder(nn.Module):
    def __init__(self, depth_in_channels, requires_grad=True):
        super().__init__()

        self.depth_conv = GatingConv2d(1, depth_in_channels[0], requires_grad=requires_grad, norm_cfg=None)

        self.downs = [DownSample(depth_in_channels[i - 1], depth_in_channels[i], conv=GatingConv2d, requires_grad=requires_grad) for i in range(1, len(depth_in_channels))]
        self.downs = nn.ModuleList(self.downs)

    def forward(self, depth):
        x = self.depth_conv(depth)

        dfeats = []
        for dc in self.downs:
            x = dc(x)
            dfeats.append(x)
        dfeats = dfeats[::-1]

        return dfeats

    def extract_feats(self, x):
        dfeats = [x]
        dfeats.extend(dc(x) for dc in self.downs)

        return dfeats[::-1]


class BaseFusion(nn.Module):
    def __init__(self, depth_encoder, mode='gating', act=nn.ReLU(inplace=True), alpha_in_channels=None, requires_grad=True, n_feats=64, n_resblocks=8, mask_channels=16):
        if alpha_in_channels is None:
            alpha_in_channels = [48, 32, 56, 160, 448]

        super().__init__()

        reverse_alpha_in_channels = alpha_in_channels[::-1] + [n_feats]

        self.depth_nets = depth_encoder

        self.depth_conv = Resnet(1, n_feats, 3, n_resblocks, n_feats, act)

        self.alphas = nn.ModuleList([DynamicConv2d(i, i, norm_cfg=None, act=act, requires_grad=requires_grad) for i in alpha_in_channels[::-1]])
        self.betas = nn.ModuleList([DynamicConv2d(i, i, norm_cfg=None, act=act, requires_grad=requires_grad) for i in alpha_in_channels[::-1]])

        depth_in_channels = [n_feats, *alpha_in_channels[:-1]]
        self.downs = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad, down_size=k!=0) for k, (i, o) in enumerate(zip(depth_in_channels, alpha_in_channels))])
        self.ups = nn.ModuleList([ConvBlock(i, o, down_size=False, requires_grad=requires_grad) for (i, o) in zip(reverse_alpha_in_channels[:-1], reverse_alpha_in_channels[1:])])

        self.n_output = 1

        self.out_net = DynamicConv2d(n_feats, self.n_output, norm_cfg=None, act=act, requires_grad=requires_grad)
        self.upscale = Upscale(mode='bilinear')

        self.masks = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad, down_size=k!=0) for k, (i, o) in enumerate(zip(depth_in_channels, alpha_in_channels))])
        self.mask_conv = Resnet(1, n_feats, 3, n_resblocks, n_feats, act, tail=True, requires_grad=requires_grad)

        self.mode = mode

    def compute_upscaled_feats(self, feats, guidances, height, width):
        upscaled_feats = feats[0]
        for i, (alpha_conv, beta_conv, up_conv) in enumerate(zip(self.alphas, self.betas, self.ups)):
            alpha = alpha_conv(guidances[i])
            beta = beta_conv(guidances[i])

            if i != len(self.alphas) - 1:
                upscaled_feats = self.upscale(upscaled_feats * alpha + beta, size=(feats[i+1].shape[2], feats[i+1].shape[3]))
                upscaled_feats = up_conv(upscaled_feats) + feats[i+1]
            else:
                upscaled_feats = self.upscale(upscaled_feats * alpha + beta, size=(height, width))
                upscaled_feats = up_conv(upscaled_feats)

        return upscaled_feats

    def compute_down_feats(self, shallow_feats):
        feats = []
        down_feat = shallow_feats
        for down_conv in self.downs:
            down_feat = down_conv(down_feat)
            feats.append(down_feat)
        return feats[::-1]

    def compute_mask_feats(self, mask):
        feats = []
        mask_feat = self.mask_conv(mask)
        for mask_conv in self.masks:
            mask_feat = mask_conv(mask_feat)
            feats.append(mask_feat)
        return feats[::-1]

    def forward(self, color_feats, depth):
        _, _, height, width = depth.size()

        mask = (depth > 0).float()

        shallow_feats = self.depth_conv(depth)
        depth_feats = self.compute_down_feats(shallow_feats)

        mask_feats = self.compute_mask_feats(mask)
        guidance_feats = [color_feats[i] * mask_feats[i] for i in range(len(mask_feats))]

        up_feats = shallow_feats + self.compute_upscaled_feats(depth_feats, guidance_feats, height, width)

        return [self.out_net(up_feats)], up_feats
