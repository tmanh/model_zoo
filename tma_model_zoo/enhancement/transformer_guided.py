import time
import torch
import torch.nn as nn
import torch.nn.functional as functional

from mmcv.runner import load_checkpoint

from ..universal.resnet import Resnet
from ..enhancement.base import ConvBlock
from ..basics.upsampling import Upscale
from ..basics.dynamic_conv import DynamicConv2d, UpSampleResidual

from .cspn_fusion import BaseCSPNFusion, AdaptiveCSPNFusion

from ..monocular_depth.depthformer import build_depther_from
from ..utils.misc import freeze_module

class TransformerGuided(nn.Module):
    # model: rgb-m, rgbm
    # refine: None, 'cspn-a', 'cspn-l'
    def __init__(self, n_feats=64, mask_channels=16, n_resblocks=8, act=nn.GELU(), model='rgb-m', refine='cspn-a', requires_grad=True):
        super().__init__()

        self.min_d = 0.0
        self.max_d = 20.0

        self.mode = 'refine'  # estimate, complete, refine
        self.model = model

        self.flag = True
        if self.flag:
            self.depth_from_color = build_depther_from('/scratch/antruong/workspace/myspace/model_zoo/tma_model_zoo/universal/configs/depthformer/depthformer_swint_w7_nyu.py').cuda()  # DepthFormerSwin()
            # load_checkpoint(self.depth_from_color, '/scratch/antruong/workspace/myspace/model_zoo/pretrained/depthformer_swint_w7_nyu.pth', map_location='cpu')
        else:
            self.depth_from_color = build_depther_from('/scratch/antruong/workspace/myspace/model_zoo/tma_model_zoo/universal/configs/binsformer/binsformer_swint_w7_nyu.py').cuda()  # DepthFormerSwin()
            # load_checkpoint(self.depth_from_color, '/scratch/antruong/workspace/myspace/model_zoo/pretrained/binsformer_swint_nyu_converted.pth', map_location='cpu')
        self.depth_from_color.min_depth = self.min_d
        self.depth_from_color.max_depth = self.max_d

        if self.mode != 'estimate':
            for param in self.depth_from_color.parameters():
                param.requires_grad = False
            self.depth_from_color.eval()

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

        if self.mode != 'complete':
            # TODO: check if freeze module is working properly before using it as the default freeze module function
            # freeze_module(self.depth_conv)

            for param in self.depth_conv.parameters():
                param.requires_grad = False
            self.depth_conv.eval()

            for param in self.downs.parameters():
                param.requires_grad = False
            self.downs.eval()

            for param in self.alphas.parameters():
                param.requires_grad = False
            self.alphas.eval()

            for param in self.betas.parameters():
                param.requires_grad = False
            self.betas.eval()

            for param in self.ups.parameters():
                param.requires_grad = False
            self.ups.eval()

            for param in self.out_net.parameters():
                param.requires_grad = False
            self.out_net.eval()

            if 'rgb-m' in self.model:
                for param in self.masks.parameters():
                    param.requires_grad = False
                self.masks.eval()

                for param in self.mask_conv.parameters():
                    param.requires_grad = False
                self.mask_conv.eval()
            else:
                for param in self.stem.parameters():
                    param.requires_grad = False
                self.stem.eval()

        self.cspn = None
        if refine == 'cspn-a':
            self.cspn = BaseCSPNFusion()
        elif refine == 'cspn-l':
            self.cspn = AdaptiveCSPNFusion()

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
        # cfeats = self.depth_from_color.extract_feats(color_lr)[::-1]
        completed, dfeat = self.fuse(cfeats, depth_lr)
        return completed, estimated, dfeat

    def forward(self, depth_lr, depth_bicubic, color_lr, mask_lr):
        start = time.time()
        if self.mode == 'estimate':
            estimated = self.estimate(color_lr)
            return None, estimated, time.time() - start

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

        return [completed], [functional.interpolate(estimated, size=(color_lr.shape[-2:]), align_corners=False, mode='bilinear')], time.time() - start
