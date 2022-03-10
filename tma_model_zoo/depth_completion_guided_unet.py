from collections import namedtuple
import sys
import torch
import torch.nn as nn

from tma_model_zoo.basics.dynamic_conv import DynamicConv2d
from tma_model_zoo.basics.upsampling import Upsample
from tma_model_zoo.efficient import EfficientNet
from tma_model_zoo.resnet import Resnet


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, act=nn.LeakyReLU(inplace=True), down_size=True):
        super().__init__()

        self.conv1 = DynamicConv2d(input_channel, output_channel, 3, act=act, batch_norm=False)
        self.conv2 = DynamicConv2d(output_channel, output_channel, 3, act=act, batch_norm=False)

        if down_size:
            self.conv3 = DynamicConv2d(output_channel, output_channel, 3, stride=2, act=act, batch_norm=False)
        else:
            self.conv3 = DynamicConv2d(output_channel, output_channel, 3, act=act, batch_norm=False)

        self.down_size = down_size

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class FusionBlock(nn.Module):
    def __init__(self, n_feats):
        super(FusionBlock, self).__init__()

        self.mask_4_depth = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False)
        self.mask_4_color = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False)
        self.alpha_conv = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False)
        self.beta_conv = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False)

    def forward(self, depth_feats, color_feats, mask_feats):
        attention_depth_feats = self.mask_4_depth(mask_feats) * depth_feats
        attention_color_feats = self.mask_4_color(mask_feats) * color_feats

        guided_a = self.alpha_conv(attention_color_feats)
        guided_b = self.beta_conv(attention_color_feats)

        residual_depth_feats = guided_a * attention_depth_feats + guided_b

        return depth_feats + residual_depth_feats


class UnetDownBLock(nn.Module):
    def __init__(self, enc_in_channels=None, enc_out_channels=None):
        if enc_in_channels is None:
            enc_in_channels = [3, 16, 32, 64, 128, 256]
        if enc_out_channels is None:
            enc_out_channels = [16, 32, 64, 128, 256, 512]
        super().__init__()

        self.encoders = [
            ConvBlock(in_channel, out_channel, down_size=i != 0)
            for i, (in_channel, out_channel) in enumerate(
                zip(enc_in_channels, enc_out_channels)
            )
        ]

        self.encoders = nn.ModuleList(*self.encoders)

    def forward(self, x):
        list_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            list_outputs.append(x)
        return list_outputs


class UnetAttentionDownBLock(nn.Module):
    def __init__(self, enc_in_channels = None, enc_out_channels = None, att_in_channels = None):
        if enc_in_channels is None:
            enc_in_channels = [1, 16, 32, 64, 128, 256]
        if enc_out_channels is None:
            enc_out_channels = [16, 32, 64, 128, 256, 512]
        if att_in_channels is None:
            att_in_channels = [16, 32, 64, 128, 256, 512]
        super().__init__()

        self.encoders = [
            ConvBlock(in_channel, out_channel, down_size=i != 0)
            for i, (in_channel, out_channel) in enumerate(
                zip(enc_in_channels, enc_out_channels)
            )
        ]

        self.encoders = nn.ModuleList(*self.encoders)

        self.attentions = [
            DynamicConv2d(
                in_channel,
                out_channel,
                batch_norm=False,
                act=nn.LeakyReLU(inplace=True),
            )
            for in_channel, out_channel in zip(att_in_channels, enc_out_channels)
        ]

        self.attentions = nn.ModuleList(*self.attentions)

    def forward(self, x, masks):
        list_outputs = []
        for encoder, attention, mask in zip(self.encoders, self.attentions, masks):
            x = encoder(x) * attention(mask)
            list_outputs.append(x)
        return list_outputs


class GuidedUnet(nn.Module):
    def __init__(self, color_enc_in_channels = None, depth_enc_in_channels = None, enc_out_channels = None):
        if color_enc_in_channels is None:
            color_enc_in_channels = [3, 16, 32, 64, 128, 256]
        if depth_enc_in_channels is None:
            depth_enc_in_channels = [1, 16, 32, 64, 128, 256]
        if enc_out_channels is None:
            enc_out_channels = [16, 32, 64, 128, 256, 512]
        super().__init__()

        self.fusions = nn.ModuleList(*[FusionBlock(n_feats) for n_feats in enc_out_channels])

        self.color_conv = UnetDownBLock(enc_in_channels=color_enc_in_channels, enc_out_channels=enc_out_channels)
        self.color_mid_conv = self.create_bottleneck()

        self.depth_conv = UnetAttentionDownBLock()
        self.depth_mid_conv = self.create_bottleneck()

        up_in_channels = enc_out_channels[1:]
        up_out_channels = depth_enc_in_channels[1:]
        self.depth_up = nn.ModuleList(*[ConvBlock(in_channel, out_channels, down_size=False)
                                        for in_channel, out_channels in zip(up_in_channels, up_out_channels)])

        # mask
        self.mask_conv = UnetDownBLock(enc_in_channels=depth_enc_in_channels, enc_out_channels=enc_out_channels)

        self.feature_net = DynamicConv2d(16, 64, 3, act=None, batch_norm=False)
        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_conv2 = nn.Conv2d(16, 1, 1, padding=0)
        self.act = nn.ReLU()

        self.upscale = Upsample()

    def create_bottleneck(self):
        return nn.Sequential(*[DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), batch_norm=False),
                               DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), batch_norm=False),
                               DynamicConv2d(512, 512, 3, act=nn.LeakyReLU(inplace=True), batch_norm=False)])

    def forward(self, color, depth, mask):
        m_feats = self.mask_conv(mask)
        c_feats_1, c_feats_2, c_feats_3, c_feats_4, c_feats_5, c_feats_6 = self.color_conv(color)
        d_feats_1, d_feats_2, d_feats_3, d_feats_4, d_feats_5, d_feats_6 = self.depth_conv(depth, m_feats)

        c_feats_8, d_feats_8 = self.bottleneck(c_feats_6, d_feats_6)

        c_feats = [c_feats_1, c_feats_2, c_feats_3, c_feats_4, c_feats_5, c_feats_8]
        d_feats = [d_feats_1, d_feats_2, d_feats_3, d_feats_4, d_feats_5, d_feats_8]
        d_decode = self.decode(c_feats, d_feats, m_feats)

        d_out = self.last_conv2(self.act(self.last_conv1(d_decode)) + d_decode)
        d_feat_out = self.feature_net(d_out)

        return d_out, d_feat_out

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
    def __init__(self, n_feats, act, mode, backbone='efficientnet-b4', enc_in_channels = None, mask_channels=16, n_resblocks=8):
        if enc_in_channels is None:
            enc_in_channels = [48, 32, 56, 160, 448]
        super().__init__()

        self.mode = mode
        self.backbone = StageEfficientNet.from_pretrained(backbone, in_channels=4 if self.mode == 'efficient-rgbm' else 3)

        self.depth_conv = Resnet(1, n_feats, 3, n_resblocks, n_feats, act)

        self.alphas = nn.ModuleList([DynamicConv2d(i, n_feats, batch_norm=False, act=act) for i in enc_in_channels][::-1])
        self.betas = nn.ModuleList([DynamicConv2d(i, n_feats, batch_norm=False, act=act) for i in enc_in_channels][::-1])
        self.downs = nn.ModuleList([ConvBlock(n_feats, n_feats) for _ in enc_in_channels])
        self.ups = nn.ModuleList([ConvBlock(n_feats, n_feats, down_size=False) for _ in enc_in_channels])

        self.n_output = 1
        if 'u' in self.mode:
            self.n_output = 64
            self.min_d, self.max_d = 0.5, 15
            self.softmax = nn.Softmax(dim=1)

        self.out_net = DynamicConv2d(n_feats, self.n_output, batch_norm=False, act=act)
        self.upscale = Upsample()

        if 'efficient-rgb-m' in self.mode:
            mask_in_channels = [mask_channels, *enc_in_channels[:-1]]
            self.masks = nn.ModuleList([ConvBlock(i, o) for i, o in zip(mask_in_channels, enc_in_channels)])
            self.mask_conv = Resnet(1, n_feats, 3, n_resblocks, mask_channels, act, tail=True)

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
        n_samples, _, height, width = depth.size()

        shallow_feats = self.depth_conv(depth)
        depth_feats = self.compute_down_feats(shallow_feats)

        if 'efficient-rgb-m' in self.mode:
            guidance_feats = self.backbone(color)[::-1]
            mask_feats = self.compute_mask_feats(mask)
            guidance_feats = [guidance_feats[i] * mask_feats[i] for i in range(len(mask_feats))]
        else:
            guidance_feats = self.backbone(torch.cat([color, mask], dim=1))[::-1]

        up_feats = shallow_feats + self.compute_upscaled_feats(depth_feats, guidance_feats, height, width)
        out = self.out_net(up_feats)

        if 'u' in self.mode:
            list_depth = torch.linspace(self.min_d, self.max_d, self.n_output, device=out.device).view(1, self.n_output, 1, 1)
            list_depth = list_depth.repeat(n_samples, 1, height, width)
            out = torch.sum(self.softmax(out) * list_depth, dim=1, keepdims=True)

        return out, up_feats


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
                        in_channels=3, num_classes=1000, **override_params):
        model = super().from_pretrained(model_name, weights_path, advprop, in_channels, num_classes, *override_params)
        model.update_stages(model_name)
        return model

    @property
    def feature_channels(self):
        if self.multi_scale_output:
            return tuple(x.num_channels for x in self.stage_specs)
        return self.stage_specs[-1].num_channels

    def forward(self, x):
        x = self._swish(self._bn0(self._conv_stem(x)))
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
