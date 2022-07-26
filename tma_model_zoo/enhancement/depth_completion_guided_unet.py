from collections import namedtuple
import sys
import time
import torch
import torch.nn as nn



from ..basics.dynamic_conv import DynamicConv2d
from ..basics.upsampling import Upscale
from ..universal.efficient import EfficientNet
from ..universal.hahi import HAHIHetero
from ..universal.resnet import Resnet
from .base import *


class GuidedUnetBasic(nn.Module):
    def __init__(self, color_enc_in_channels=None, depth_enc_in_channels=None, enc_out_channels=None, requires_grad=True):
        if color_enc_in_channels is None:
            color_enc_in_channels = [4, 16, 32, 64, 128, 256]
        if depth_enc_in_channels is None:
            depth_enc_in_channels = [1, 16, 32, 64, 128, 256]
        if enc_out_channels is None:
            enc_out_channels = [16, 32, 64, 128, 256, 512]
        super().__init__()

        self.fusions = nn.ModuleList([FusionBlock(n_feats, requires_grad=requires_grad) for n_feats in enc_out_channels])

        self.color_conv = UnetDownBLock(enc_in_channels=color_enc_in_channels, enc_out_channels=enc_out_channels, requires_grad=requires_grad)
        self.color_mid_conv = self.create_bottleneck(requires_grad=requires_grad)

        self.depth_conv = UnetDownBLock(enc_in_channels=depth_enc_in_channels, enc_out_channels=enc_out_channels, requires_grad=requires_grad)
        self.depth_mid_conv = self.create_bottleneck(requires_grad=requires_grad)

        up_in_channels = enc_out_channels[1:]
        up_out_channels = depth_enc_in_channels[1:]
        self.depth_up = nn.ModuleList([ConvBlock(in_channel, out_channels, down_size=False, requires_grad=requires_grad)
                                        for in_channel, out_channels in zip(up_in_channels, up_out_channels)])

        # mask
        self.mask_conv = UnetDownBLock(enc_in_channels=depth_enc_in_channels, enc_out_channels=enc_out_channels, requires_grad=requires_grad)

        self.last_conv1 = DynamicConv2d(16, 16, 3, norm_cfg=None, requires_grad=requires_grad)
        self.last_conv2 = DynamicConv2d(16, 1, 1, norm_cfg=None, act=None, requires_grad=requires_grad)
        self.act = nn.ReLU()

        self.upscale = Upscale(mode='bilinear')

    def create_bottleneck(self, requires_grad=True):
        return nn.Sequential(*[DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), norm_cfg=None, requires_grad=requires_grad),
                               DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), norm_cfg=None, requires_grad=requires_grad),
                               DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), norm_cfg=None, requires_grad=requires_grad)])

    def forward(self, color, depth, mask):
        start = time.time()
        guidance = torch.cat([color, mask], dim=1)
        c_feats_1, c_feats_2, c_feats_3, c_feats_4, c_feats_5, c_feats_6 = self.color_conv(guidance)
        d_feats_1, d_feats_2, d_feats_3, d_feats_4, d_feats_5, d_feats_6 = self.depth_conv(depth)

        c_feats_8, d_feats_8 = self.bottleneck(c_feats_6, d_feats_6)

        c_feats = [c_feats_1, c_feats_2, c_feats_3, c_feats_4, c_feats_5, c_feats_8]
        d_feats = [d_feats_1, d_feats_2, d_feats_3, d_feats_4, d_feats_5, d_feats_8]
        d_decode = self.decode(c_feats, d_feats)

        d_out = self.last_conv2(self.act(self.last_conv1(d_decode)) + d_decode)

        return [d_out], d_decode, time.time() - start

    def bottleneck(self, c_feats_6, d_feats_6):
        c_feats_7 = self.color_mid_conv(c_feats_6)
        c_feats_8 = c_feats_7 + c_feats_6

        d_feats_7 = self.depth_mid_conv(d_feats_6)
        d_feats_8 = d_feats_7 + d_feats_6
        return c_feats_8, d_feats_8

    def decode(self, c_feats, d_feats):
        x_up = 0
        for i in range(len(c_feats)):
            x_up = self.fusions[-i-1](x_up + d_feats[-i-1], c_feats[-i-1], None)

            if i != len(c_feats) - 1:
                x_up = self.upscale(x_up, size=(d_feats[-i-2].shape[2], d_feats[-i-2].shape[3]))
                x_up = self.depth_up[-i-1](x_up)

        return x_up



class GuidedUnet(nn.Module):
    def __init__(self, color_enc_in_channels=None, depth_enc_in_channels=None, enc_out_channels=None, requires_grad=True, mask_channels=16):
        if color_enc_in_channels is None:
            color_enc_in_channels = [3, 16, 32, 64, 128, 256]
        if depth_enc_in_channels is None:
            depth_enc_in_channels = [1, 16, 32, 64, 128, 256]
        if enc_out_channels is None:
            enc_out_channels = [16, 32, 64, 128, 256, 512]
        super().__init__()

        self.fusions = nn.ModuleList([FusionBlock(n_feats, requires_grad=requires_grad) for n_feats in enc_out_channels])

        self.color_conv = UnetDownBLock(enc_in_channels=color_enc_in_channels, enc_out_channels=enc_out_channels, requires_grad=requires_grad)
        self.color_mid_conv = self.create_bottleneck(requires_grad=requires_grad)

        self.depth_conv = UnetAttentionDownBLock(requires_grad=requires_grad)
        self.depth_mid_conv = self.create_bottleneck(requires_grad=requires_grad)

        up_in_channels = enc_out_channels[1:]
        up_out_channels = depth_enc_in_channels[1:]
        self.depth_up = nn.ModuleList([ConvBlock(in_channel, out_channels, down_size=False, requires_grad=requires_grad)
                                        for in_channel, out_channels in zip(up_in_channels, up_out_channels)])

        # mask
        self.mask_conv = UnetDownBLock(enc_in_channels=depth_enc_in_channels, enc_out_channels=enc_out_channels, requires_grad=requires_grad)

        self.last_conv1 = DynamicConv2d(16, 16, 3, norm_cfg=None, requires_grad=requires_grad)
        self.last_conv2 = DynamicConv2d(16, 1, 1, norm_cfg=None, act=None, requires_grad=requires_grad)
        self.act = nn.ReLU()

        self.upscale = Upscale(mode='bilinear')

    def create_bottleneck(self, requires_grad=True):
        return nn.Sequential(*[DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), norm_cfg=None, requires_grad=requires_grad),
                               DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), norm_cfg=None, requires_grad=requires_grad),
                               DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), norm_cfg=None, requires_grad=requires_grad)])

    def forward(self, color, depth, mask):
        start = time.time()
        m_feats = self.mask_conv(mask)
        c_feats_1, c_feats_2, c_feats_3, c_feats_4, c_feats_5, c_feats_6 = self.color_conv(color)
        d_feats_1, d_feats_2, d_feats_3, d_feats_4, d_feats_5, d_feats_6 = self.depth_conv(depth, m_feats)

        c_feats_8, d_feats_8 = self.bottleneck(c_feats_6, d_feats_6)

        c_feats = [c_feats_1, c_feats_2, c_feats_3, c_feats_4, c_feats_5, c_feats_8]
        d_feats = [d_feats_1, d_feats_2, d_feats_3, d_feats_4, d_feats_5, d_feats_8]
        d_decode = self.decode(c_feats, d_feats, m_feats)

        d_out = self.last_conv2(self.act(self.last_conv1(d_decode)) + d_decode)

        return [d_out], d_decode, time.time() - start

    def bottleneck(self, c_feats_6, d_feats_6):
        c_feats_7 = self.color_mid_conv(c_feats_6)
        c_feats_8 = c_feats_7 + c_feats_6

        d_feats_7 = self.depth_mid_conv(d_feats_6)
        d_feats_8 = d_feats_7 + d_feats_6
        return c_feats_8, d_feats_8

    def decode(self, c_feats, d_feats, m_feats):
        x_up = 0
        for i in range(len(m_feats)):
            x_up = self.fusions[-i-1](x_up + d_feats[-i-1], c_feats[-i-1], m_feats[-i-1])

            if i != len(m_feats) - 1:
                x_up = self.upscale(x_up, size=(d_feats[-i-2].shape[2], d_feats[-i-2].shape[3]))
                x_up = self.depth_up[-i-1](x_up)

        return x_up


class GuidedEfficientNet(nn.Module):
    def __init__(self, n_feats=64, act=nn.ReLU(inplace=True), mode='rgb-m', backbone='efficientnet-b4', enc_in_channels = None, mask_channels=16, n_resblocks=8, requires_grad=True, neck=True):
        if enc_in_channels is None:
            enc_in_channels = [48, 32, 56, 160, 448]
        super().__init__()

        self.mode = mode
        self.backbone = StageEfficientNet.from_pretrained(backbone, in_channels=4 if 'rgbm' in self.mode else 3, requires_grad=requires_grad)

        self.depth_conv = Resnet(1, n_feats, 3, n_resblocks, n_feats, act)

        self.alphas = nn.ModuleList([DynamicConv2d(i, n_feats, norm_cfg=None, act=act, requires_grad=requires_grad) for i in enc_in_channels][::-1])
        self.betas = nn.ModuleList([DynamicConv2d(i, n_feats, norm_cfg=None, act=act, requires_grad=requires_grad) for i in enc_in_channels][::-1])
        self.downs = nn.ModuleList([ConvBlock(n_feats, n_feats, requires_grad=requires_grad) for _ in enc_in_channels])
        self.ups = nn.ModuleList([ConvBlock(n_feats, n_feats, down_size=False, requires_grad=requires_grad) for _ in enc_in_channels])

        self.list_feats = enc_in_channels[::-1]
        if neck:
            self.neck = HAHIHetero(in_channels=self.list_feats, out_channels=self.list_feats, embedding_dim=256, num_feature_levels=len(self.list_feats), requires_grad=requires_grad)

        self.n_output = 1

        self.out_net = DynamicConv2d(n_feats, self.n_output, norm_cfg=None, act=act, requires_grad=requires_grad)
        self.upscale = Upscale(mode='bilinear')

        if 'rgb-m' in self.mode:
            mask_in_channels = [mask_channels, *enc_in_channels[:-1]]
            self.masks = nn.ModuleList([ConvBlock(i, o, requires_grad=requires_grad) for i, o in zip(mask_in_channels, enc_in_channels)])
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

    def forward(self, color, depth, mask):
        start = time.time()
        _, _, height, width = depth.size()

        shallow_feats = self.depth_conv(depth)
        depth_feats = self.compute_down_feats(shallow_feats)

        if 'rgb-m' in self.mode:
            guidance_feats = self.backbone(color)[::-1]
            if self.neck:
                guidance_feats = self.neck(guidance_feats)
            mask_feats = self.compute_mask_feats(mask)
            guidance_feats = [guidance_feats[i] * mask_feats[i] for i in range(len(mask_feats))]
        else:
            guidance_feats = self.backbone(torch.cat([color, depth], dim=1))[::-1]
            if self.neck:
                guidance_feats = self.neck(guidance_feats)

        up_feats = shallow_feats + self.compute_upscaled_feats(depth_feats, guidance_feats, height, width)
        out = self.out_net(up_feats)

        if 'residual' in self.mode:
            out = out + depth

        return [out], up_feats, time.time()-start


StageSpec = namedtuple("StageSpec", ["num_channels", "stage_stamp"],)

efficientnet_b0 = ((24, 3), (40, 4), (112, 9), (320, 16))
efficientnet_b0 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b0)

efficientnet_b1 = ((24, 5), (40, 8), (112, 16), (320, 23))
efficientnet_b1 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b1)

efficientnet_b2 = ((24, 5), (48, 8), (120, 16), (352, 23))
efficientnet_b2 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b2)

efficientnet_b3 = ((32, 5), (48, 8), (136, 18), (384, 26))
efficientnet_b3 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b3)

efficientnet_b4 = ((32, 6), (56, 10), (160, 22), (448, 32))
efficientnet_b4 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b4)

efficientnet_b5 = ((40, 8), (64, 13), (176, 27), (512, 39))
efficientnet_b5 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b5)

efficientnet_b6 = ((40, 9), (72, 15), (200, 31), (576, 45))
efficientnet_b6 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b6)

efficientnet_b7 = ((48, 11), (80, 18), (224, 38), (640, 55))
efficientnet_b7 = tuple(StageSpec(num_channels=nc, stage_stamp=ss) for (nc, ss) in efficientnet_b7)


class StageEfficientNet(EfficientNet):
    def update_stages(self, model_name):
        self.multi_scale_output = True
        self.stage_specs = sys.modules[__name__].__getattribute__(model_name.replace('-', '_'))
        self.num_blocks = len(self._blocks)

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, requires_grad=True, **override_params):
        model = super().from_pretrained(model_name, weights_path, advprop, in_channels, num_classes, requires_grad=requires_grad, *override_params)
        model.update_stages(model_name)
        return model

    @property
    def feature_channels(self):
        if self.multi_scale_output:
            return tuple(x.num_channels for x in self.stage_specs)
        return self.stage_specs[-1].num_channels

    def forward(self, x):
        x = self._conv_stem(x)
        block_idx = 0.
        features = [x]
        for stage in [
            self._blocks[:self.stage_specs[0].stage_stamp],
            self._blocks[self.stage_specs[0].stage_stamp:self.stage_specs[1].stage_stamp],
            self._blocks[self.stage_specs[1].stage_stamp:self.stage_specs[2].stage_stamp],
            self._blocks[self.stage_specs[2].stage_stamp:],
        ]:
            for block in stage:
                x = block(x, self._global_params.drop_connect_rate * block_idx / self.num_blocks)
                block_idx += 1.

            features.append(x)

        if self.multi_scale_output:
            return features
        return [x]
