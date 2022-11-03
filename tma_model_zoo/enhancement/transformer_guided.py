import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional

from mmcv.runner import load_checkpoint

from ..universal.resnet import Resnet
from ..enhancement.base import ConvBlock
from ..basics.upsampling import Upscale
from ..basics.dynamic_conv import DynamicConv2d, UpSampleResidual

from .cspn_fusion import BaseCSPNFusion

from ..monocular_depth.depthformer import build_depther_from
from ..utils.misc import freeze_module


class TransformerGuided(nn.Module):
    def load_estimator(self, estimator_type):
        config_path = os.path.abspath(__file__).replace('enhancement/transformer_guided.py', 'universal/configs/')

        if estimator_type == 'binsformer_swint_nyu_converted':
            config_path = os.path.join(config_path, 'binsformer/binsformer_swint_w7_nyu.py')
        elif estimator_type == 'depthformer_swint_w7_nyu':
            config_path = os.path.join(config_path, 'depthformer/depthformer_swint_w7_nyu.py')
            
        self.depth_from_color = build_depther_from(config_path)

        pretrained_path = '/scratch/antruong/workspace/myspace/model_zoo/pretrained'
        pretrained_path = os.path.join(pretrained_path, f'{estimator_type}.pth')
        load_checkpoint(self.depth_from_color, pretrained_path, map_location='cpu')

    def freeze(self, pretrained_estimator):
        if pretrained_estimator:
            freeze_module(self.depth_from_color)

        if 'estimate' not in self.modes:
            freeze_module(self.depth_from_color)

        if 'complete' not in self.modes:
            freeze_module(self.depth_conv)
            freeze_module(self.downs)

            freeze_module(self.alphas)
            freeze_module(self.betas)
            freeze_module(self.ups)

            freeze_module(self.out_net)

            if 'rgb-m' in self.model:
                freeze_module(self.masks)
                freeze_module(self.mask_conv)
            else:
                freeze_module(self.stem)

        if 'refine' not in self.modes and self.cspn is not None:
            freeze_module(self.cspn)


    # model: rgb-m, rgbm
    # modes: estimate, complete, refine
    # refine: None, 'cspn-a'
    def __init__(self, n_feats=64, mask_channels=16, n_resblocks=8, act=nn.GELU(), model='rgb-m', refine='cspn-a', requires_grad=True,
                 min_d=0.0, max_d=20.0, estimator_type='depthformer_swint_w7_nyu', pretrained_estimator=False, modes=None):
        super().__init__()

        self.modes = ['refine'] if modes is None else modes
        self.model = model

        self.load_estimator(estimator_type)
        self.set_depth_range(min_d=min_d, max_d=max_d)

        self.list_feats = self.depth_from_color.list_feats
        self.level2full = self.depth_from_color.level2full

        if self.level2full == 0:
            self.depth_conv = Resnet(1, n_feats, 3, n_resblocks, n_feats, act, requires_grad=requires_grad)
            depth_in_channels = [n_feats, *self.list_feats]
            self.downs = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad, down_size=k>0) for k, (i, o) in enumerate(zip(depth_in_channels[:-1], depth_in_channels[1:]))])
        elif self.level2full == 1:
            self.depth_conv = Resnet(1, n_feats, 3, n_resblocks, n_feats, act, requires_grad=requires_grad)
            depth_in_channels = [n_feats, *self.list_feats]
            self.downs = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad, down_size=True) for i, o in zip(depth_in_channels[:-1], depth_in_channels[1:])])
        else:
            self.depth_conv = Resnet(1, n_feats // 2, 3, n_resblocks, n_feats // 2, act, requires_grad=requires_grad)
            depth_in_channels = [n_feats // 2, n_feats, *self.list_feats]
            self.downs = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad, down_size=True) for i, o in zip(depth_in_channels[:-1], depth_in_channels[1:])])

        self.alphas = nn.ModuleList([DynamicConv2d(i, i, norm_cfg=None, act=act, requires_grad=requires_grad) for i in self.list_feats][::-1])
        self.betas = nn.ModuleList([DynamicConv2d(i, i, norm_cfg=None, act=act, requires_grad=requires_grad) for i in self.list_feats][::-1])
        self.ups = nn.ModuleList([UpSampleResidual(i, o, requires_grad=requires_grad, act=nn.GELU()) for i, o in zip(depth_in_channels[::-1][:-1], depth_in_channels[::-1][1:])])

        self.n_output = 1

        self.out_net = DynamicConv2d(n_feats, self.n_output, norm_cfg=None, act=None, requires_grad=requires_grad)
        self.upscale = Upscale(mode='bilinear')

        if 'rgb-m' in self.model:
            mask_in_channels = [mask_channels, *self.list_feats[:-1]]
            self.masks = nn.ModuleList([ConvBlock(i, o, down_size=False, requires_grad=requires_grad) for i, o in zip(mask_in_channels, self.list_feats)])
            self.mask_conv = Resnet(1, n_feats, 3, n_resblocks, mask_channels, act, tail=True, requires_grad=requires_grad)
        else:
            self.stem = DynamicConv2d(4, 3, norm_cfg=None, act=nn.GELU(), requires_grad=requires_grad)

        self.cspn = None
        if refine == 'cspn-a':
            self.cspn = BaseCSPNFusion()

        self.freeze(pretrained_estimator)

    def set_depth_range(self, min_d, max_d):
        self.depth_from_color.set_depth_range(min_d, max_d)

    def compute_upscaled_feats(self, feats, guidances, height, width):
        upscaled_feats = feats[0]
        for i, (alpha_conv, beta_conv, up_conv) in enumerate(zip(self.alphas, self.betas, self.ups)):
            alpha = alpha_conv(guidances[i])
            beta = beta_conv(guidances[i])

            if i != len(self.alphas) - 1:
                upscaled_feats = upscaled_feats * alpha + beta
                upscaled_feats = up_conv(upscaled_feats, feats[i+1])
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

    def compute_mask_feats(self, mask, resolutions):
        feats = []
        mask_feat = self.mask_conv(mask)

        for mask_conv, resolution in zip(self.masks, resolutions):
            mask_feat = functional.interpolate(mask_feat, size=resolution, align_corners=False, mode='bilinear')
            mask_feat = mask_conv(mask_feat)
            feats.append(mask_feat)
        return feats

    def estimate(self, color_lr):
        o = self.depth_from_color.simple_run(color_lr)

        if not self.flag:
            o = o[0][-1]

        return [functional.interpolate(o, size=(color_lr.shape[-2:]), align_corners=False, mode='bilinear')]

    def extract_feats(self, depth_lr, color_lr):
        estimated, cfeats = self.depth_from_color.extract_feats(color_lr)
        completed, dfeat = self.fuse(cfeats, depth_lr)
        return completed, estimated, dfeat

    def forward(self, depth_lr, depth_bicubic, color_lr, mask_lr):
        start = time.time()
        if 'estimate' in self.modes:
            estimated = self.estimate(color_lr)
            return None, estimated, None, time.time() - start

        _, _, height, width = depth_lr.shape

        shallow_feats = self.depth_conv(depth_lr)
        depth_feats = self.compute_down_feats(shallow_feats)

        if 'rgb-m' in self.model:
            estimated, guidance_feats = self.depth_from_color.extract_feats(color_lr)
            resolutions = [g.shape[-2:] for g in guidance_feats]
            mask_feats = self.compute_mask_feats(mask_lr, resolutions)
            guidance_feats = [guidance_feats[i] * mask_feats[i] for i in range(len(mask_feats))][::-1]
        else:
            estimated, guidance_feats = self.depth_from_color.extract_feats(color_lr)
            guidance_feats = guidance_feats[::-1]

        up_feats = shallow_feats + self.compute_upscaled_feats(depth_feats, guidance_feats, height, width)
        completed = depth_lr + self.out_net(up_feats)

        if self.cspn is not None:
            completed = self.cspn(up_feats, depth_lr, completed, mask_lr)

        return [completed], [functional.interpolate(estimated, size=(color_lr.shape[-2:]), align_corners=False, mode='bilinear')], up_feats, time.time() - start
