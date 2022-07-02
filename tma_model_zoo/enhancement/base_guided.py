import torch
import torch.nn as nn
import torch.nn.functional as functional

from tma_model_zoo.basics.upsampling import Upscale
from tma_model_zoo.basics.dynamic_conv import GatingConv2d, DynamicConv2d, DownSample, UpSample


class DepthEncoder(nn.Module):
    def __init__(self, depth_in_channels, requires_grad=True):
        super().__init__()

        self.depth_conv = GatingConv2d(1, depth_in_channels[0], requires_grad=requires_grad)

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

    def extract_feats(self, depth):
        x = self.depth_conv(depth)

        dfeats = []
        for dc in self.downs:
            x = dc(x)
            dfeats.append(x)
        dfeats = dfeats[::-1]

        return dfeats


class BaseFusion(nn.Module):
    def __init__(self, depth_encoder, mode='gating', act=nn.ReLU(inplace=True), alpha_in_channels=None, requires_grad=True):
        if alpha_in_channels is None:
            alpha_in_channels = [48, 32, 56, 160, 448]

        super().__init__()

        reverse_alpha_in_channels = alpha_in_channels[::-1] + [alpha_in_channels[0] // 2]

        self.depth_nets = depth_encoder

        self.alphas = nn.ModuleList([GatingConv2d(i, i, act=act) for i in alpha_in_channels][::-1])
        self.betas = nn.ModuleList([GatingConv2d(i, i, act=act) for i in alpha_in_channels][::-1])
        self.refine = nn.ModuleList([GatingConv2d(i, i, act=act) for i in alpha_in_channels][::-1])

        self.neck = GatingConv2d(alpha_in_channels[-1], alpha_in_channels[-1], requires_grad=requires_grad)
        self.ups = [UpSample(reverse_alpha_in_channels[i] * 2, reverse_alpha_in_channels[i + 1], conv=GatingConv2d, requires_grad=requires_grad) for i in range(len(alpha_in_channels))]
        self.ups = nn.ModuleList(self.ups)

        self.out_net = DynamicConv2d(alpha_in_channels[0] // 2, 1, act=act)
        self.upscale = Upscale()

        self.mode = mode
        if 'rgb-m' in self.mode:
            self.masks = nn.ModuleList([GatingConv2d(1, i) for i in alpha_in_channels[::-1]])
        else:
            self.masks = nn.ModuleList([GatingConv2d(i, i) for i in alpha_in_channels[::-1]])

    def forward(self, color_feats, depth):
        mask = (depth > 0).float()

        dfeats = self.depth_nets.extract_feats(depth)

        x = self.neck(dfeats[0])

        for i, uc in enumerate(self.ups):
            if 'rgb-m' in self.mode:
                if mask.shape[-2:] != color_feats[i].shape[-2:]:
                    scaled = functional.interpolate(mask, size=color_feats[i].shape[-2:], mode='bilinear', align_corners=True)
                else:
                    scaled = mask

                mfeat = self.masks[i](scaled)
            else:
                mfeat = self.masks[i](dfeats[i])

            guidance_feat =  color_feats[i] * mfeat

            if 'rgb-m' in self.mode:
                alpha = self.alphas[i](guidance_feat)
            else:
                alpha = self.alphas[i](dfeats[i])
            
            beta = self.betas[i](guidance_feat)
            
            x = self.refine[i](alpha * dfeats[i] + (1 - alpha) * beta)
            x = uc(x, dfeats[i])

        return self.out_net(x), x
