import torch.nn as nn

from ..basics.dynamic_conv import DynamicConv2d


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, act=nn.LeakyReLU(inplace=True), down_size=True, requires_grad=True):
        super().__init__()

        self.conv1 = DynamicConv2d(input_channel, output_channel, 3, act=act, norm_cfg=None, requires_grad=requires_grad)
        self.conv2 = DynamicConv2d(output_channel, output_channel, 3, act=act, norm_cfg=None, requires_grad=requires_grad)

        if down_size:
            self.conv3 = DynamicConv2d(output_channel, output_channel, 3, stride=2, act=act, norm_cfg=None, requires_grad=requires_grad)
        else:
            self.conv3 = DynamicConv2d(output_channel, output_channel, 3, act=act, norm_cfg=None, requires_grad=requires_grad)

        self.down_size = down_size

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class FusionBlock(nn.Module):
    def __init__(self, n_feats, requires_grad=True):
        super(FusionBlock, self).__init__()

        self.mask_4_depth = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False, requires_grad=requires_grad)
        self.mask_4_color = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False, requires_grad=requires_grad)
        self.alpha_conv = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False, requires_grad=requires_grad)
        self.beta_conv = ConvBlock(n_feats, n_feats, act=nn.LeakyReLU(inplace=True), down_size=False, requires_grad=requires_grad)

    def forward(self, depth_feats, color_feats, mask_feats):
        if mask_feats is not None:
            attention_depth_feats = self.mask_4_depth(mask_feats) * depth_feats
            attention_color_feats = self.mask_4_color(mask_feats) * color_feats
        else:
            attention_depth_feats = depth_feats
            attention_color_feats = color_feats

        guided_a = self.alpha_conv(attention_color_feats)
        guided_b = self.beta_conv(attention_color_feats)

        residual_depth_feats = guided_a * attention_depth_feats + guided_b

        return depth_feats + residual_depth_feats


class UnetDownBLock(nn.Module):
    def __init__(self, enc_in_channels=None, enc_out_channels=None, requires_grad=True):
        if enc_in_channels is None:
            enc_in_channels = [3, 16, 32, 64, 128, 256]
        if enc_out_channels is None:
            enc_out_channels = [16, 32, 64, 128, 256, 512]
        super().__init__()

        self.encoders = [
            ConvBlock(in_channel, out_channel, down_size=i != 0, requires_grad=requires_grad)
            for i, (in_channel, out_channel) in enumerate(
                zip(enc_in_channels, enc_out_channels)
            )
        ]

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        list_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            list_outputs.append(x)
        return list_outputs


class UnetAttentionDownBLock(nn.Module):
    def __init__(self, enc_in_channels = None, enc_out_channels = None, att_in_channels = None, requires_grad=True):
        if enc_in_channels is None:
            enc_in_channels = [1, 16, 32, 64, 128, 256]
        if enc_out_channels is None:
            enc_out_channels = [16, 32, 64, 128, 256, 512]
        if att_in_channels is None:
            att_in_channels = [16, 32, 64, 128, 256, 512]
        super().__init__()

        self.encoders = [
            ConvBlock(in_channel, out_channel, down_size=i != 0, requires_grad=requires_grad)
            for i, (in_channel, out_channel) in enumerate(
                zip(enc_in_channels, enc_out_channels)
            )
        ]

        self.encoders = nn.ModuleList(self.encoders)

        self.attentions = [
            DynamicConv2d(in_channel, out_channel, norm_cfg=None, act=nn.LeakyReLU(inplace=True)) for in_channel, out_channel in zip(att_in_channels, enc_out_channels)
        ]

        self.attentions = nn.ModuleList(self.attentions)

    def forward(self, x, masks):
        list_outputs = []
        for encoder, attention, mask in zip(self.encoders, self.attentions, masks):
            x = encoder(x) * attention(mask)
            list_outputs.append(x)
        return list_outputs
