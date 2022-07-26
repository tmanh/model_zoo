import time
import torch
import torch.nn as nn


from ..universal.resnet import Resnet
from ..enhancement.base import ConvBlock
from ..basics.upsampling import Upscale
from ..basics.dynamic_conv import DynamicConv2d
from ..monocular_depth.depthformer import DepthFormer, DepthFormerEncode

from .base_guided import BaseFusion, DepthEncoder
from .cspn_fusion import CSPNFusion


class TransformerGuided(nn.Module):
    def __init__(self, n_feats=64, mask_channels=16, n_resblocks=8, act=nn.GELU(), model='rgb-m', requires_grad=True):
        super().__init__()

        self.mode = 'completion'  # estimate, completion
        self.model = model

        # self.depth_from_color = DepthFormerEncode(requires_grad=True)
        self.depth_from_color = DepthFormer(requires_grad=True)

        self.list_feats = self.depth_from_color.list_feats

        self.depth_conv = Resnet(1, n_feats, 3, n_resblocks, n_feats, act, requires_grad=requires_grad)

        depth_in_channels = [n_feats, *self.list_feats[:-1]]
        self.alphas = nn.ModuleList([DynamicConv2d(i, i, norm_cfg=None, act=act, requires_grad=requires_grad) for i in self.list_feats][::-1])
        self.betas = nn.ModuleList([DynamicConv2d(i, i, norm_cfg=None, act=act, requires_grad=requires_grad) for i in self.list_feats][::-1])
        self.downs = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad, down_size=k>0) for k, (i, o) in enumerate(zip(depth_in_channels, self.list_feats))])
        self.ups = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad, down_size=False) for k, (i, o) in enumerate(zip(self.list_feats[::-1], depth_in_channels[::-1]))])

        self.n_output = 1

        self.out_net = DynamicConv2d(n_feats, self.n_output, norm_cfg=None, act=act, requires_grad=requires_grad)
        self.upscale = Upscale(mode='bilinear')

        if 'rgb-m' in self.model:
            mask_in_channels = [mask_channels, *self.list_feats[:-1]]
            self.masks = nn.ModuleList([ConvBlock(i, o, down_size=k>0, requires_grad=requires_grad) for k, (i, o) in enumerate(zip(mask_in_channels, self.list_feats))])
            self.mask_conv = Resnet(1, n_feats, 3, n_resblocks, mask_channels, act, tail=True, requires_grad=requires_grad)

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

    def estimate(self, color_lr):
        return self.depth_from_color(color_lr)

    def extract_feats(self, depth_lr, color_lr):
        estimated, cfeats = self.depth_from_color.extract_feats(color_lr)
        # cfeats = self.depth_from_color.extract_feats(color_lr)[::-1]
        completed, dfeat = self.fuse(cfeats, depth_lr)
        return completed, estimated, dfeat

    def forward(self, depth_lr, depth_bicubic, color_lr, mask_lr):
        start = time.time()
        if self.mode == 'estimate':
            estimated = self.depth_from_color(color_lr)
            return None, estimated, time.time() - start
        else:
            _, _, height, width = depth_lr.shape

            shallow_feats = self.depth_conv(depth_lr)
            depth_feats = self.compute_down_feats(shallow_feats)

            if 'rgb-m' in self.model:
                estimated, guidance_feats = self.depth_from_color.extract_feats(color_lr)
                # guidance_feats = self.depth_from_color.extract_feats(color_lr)[::-1]
                mask_feats = self.compute_mask_feats(mask_lr)
                guidance_feats = [guidance_feats[i] * mask_feats[i] for i in range(len(mask_feats))]
            else:
                guidance_feats = self.depth_from_color(torch.cat([color_lr, depth_lr], dim=1))[::-1]

            up_feats = shallow_feats + self.compute_upscaled_feats(depth_feats, guidance_feats, height, width)
            completed = self.out_net(up_feats)

            return [completed], estimated, time.time() - start
