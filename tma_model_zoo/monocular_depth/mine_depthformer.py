from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..universal.resnet import Resnet

from ..enhancement.guided_depth_completion import StageEfficientNet

from ..basics.dynamic_conv import DynamicConv2d, UpSample, DownSample
from ..basics.norm import NormBuilder
from ..universal.swin import SwinTransformerV2
from ..universal.hahi import HAHIHetero


class EfficientEncode(nn.Module):
    def __init__(self, backbone='efficientnet-b4', requires_grad=True):
        super().__init__()

        self.list_feats = [32, 48, 32, 56, 160, 448]
        self.stem = DynamicConv2d(3, 32)
        self.backbone = StageEfficientNet.from_pretrained(backbone, in_channels=3, requires_grad=False)

        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0
                module.eval()
            for param in module.parameters():
                param.requires_grad = False

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.neck = HAHIHetero(in_channels=self.list_feats, out_channels=self.list_feats, embedding_dim=256, num_feature_levels=len(self.list_feats), requires_grad=requires_grad)
        
    def forward(self, color):
        shallow_feats = self.stem(color)
        deep_feats = self.backbone(color)
        return self.neck([shallow_feats,]+deep_feats)


class DepthFormerStem(nn.Module):
    def __init__(self, in_dim, out_dims, requires_grad=True):
        super().__init__()

        self.conv1 = DynamicConv2d(in_dim, out_dims[0], kernel_size=7, stride=1, bias=False, requires_grad=requires_grad)
        self.conv2 = DownSample(out_dims[0], out_dims[1], requires_grad=requires_grad)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1, x2


class DepthFormerEncode(nn.Module):
    def __init__(self, in_channels=3, requires_grad=True):
        super().__init__()
        self.swin_transformer = SwinTransformerV2(in_channels=in_channels, requires_grad=False)
        self.swin_transformer.load_pretrained()

        if isinstance(self.swin_transformer, SwinTransformerV2):
            self.nfeats = [32, 64]
            self.stem = DepthFormerStem(in_channels, self.nfeats, requires_grad=requires_grad)

        self.list_feats = self.swin_transformer.list_feats
        self.norms = nn.ModuleList([NormBuilder.build(cfg=dict(type='BN2d', requires_grad=requires_grad), num_features=f) for f in self.list_feats])

        if isinstance(self.swin_transformer, SwinTransformerV2):
            self.list_feats = self.nfeats + self.list_feats
        self.neck = HAHIHetero(in_channels=self.list_feats, out_channels=self.list_feats, embedding_dim=256, num_feature_levels=len(self.list_feats), requires_grad=requires_grad)

    def conv_stem(self, x, resolution):
        if x.shape[1:3] != resolution:
            x = functional.interpolate(x, size=resolution, mode='bilinear', align_corners=True)
        return self.stem(x)

    def extract_feats(self, x):
        return self.forward(x)

    def forward(self, x):
        n_samples = x.shape[0]

        outs = []
        stem_feats = self.conv_stem(x, x.shape[1:3])
        outs += stem_feats

        transformer_outs, resolutions = self.swin_transformer(x)
        self.norm_transformer_outputs(n_samples, outs, transformer_outs, resolutions)
        return self.neck(outs)

    def norm_transformer_outputs(self, n_samples, outs, transformer_outs, resolutions):
        for i in range(len(transformer_outs)):
            o = transformer_outs[i]
            r = resolutions[i]

            no = self.norms[i](o.view(n_samples, r[0], r[1], -1).permute(0, 3, 1, 2).contiguous())
            outs.append(no)


class DepthFormerDecode(nn.Module):
    min_depth = 0.01

    def __init__(self, in_channels, norm_cfg=None, act=nn.ReLU(inplace=True), requires_grad=True):
        super().__init__()

        self.in_channels = in_channels[::-1]
        self.up_sample_channels = in_channels[::-1]

        self.norm_cfg = norm_cfg
        self.act = act
        self.relu = nn.ReLU(inplace=True)

        self.conv_list = nn.ModuleList()
        self.conv_depth = nn.ModuleList()

        for index, (in_channel, up_channel) in enumerate(zip(self.in_channels, self.up_sample_channels)):
            if index == 0:
                self.conv_list.append(DynamicConv2d(in_channels=in_channel, out_channels=up_channel, kernel_size=1, stride=1, norm_cfg=None, act=None, requires_grad=requires_grad))
            else:
                self.conv_list.append(UpSample(skip_input=in_channel + up_channel_temp, output_features=up_channel, norm_cfg=self.norm_cfg, act=self.act, requires_grad=requires_grad))

            self.conv_depth.append(nn.Conv2d(up_channel, 1, kernel_size=3, padding=1, stride=1))

            # save earlier fusion target
            up_channel_temp = up_channel

        for cd in self.conv_depth:
            for param in cd.parameters():
                param.requires_grad = requires_grad

    def extract_feats(self, inputs):
        temp_feat_list = []
        for index, feat in enumerate(inputs[::-1]):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
            else:
                skip_feat = feat
                up_feat = temp_feat_list[index-1]
                temp_feat = self.conv_list[index](up_feat, skip_feat)
            temp_feat_list.append(temp_feat)

        return [self.relu(self.conv_depth[-1](temp_feat_list[-1]))], temp_feat_list

    def forward(self, inputs):
        _, temp_feat_list = self.extract_feats(inputs)
        return [self.relu(self.conv_depth[i](temp_feat_list[i])) for i in range(len(temp_feat_list))]


class DepthFormer(nn.Module):
    def __init__(self, in_channels=3, requires_grad=True):
        super().__init__()

        self.encode = DepthFormerEncode(in_channels, requires_grad=requires_grad)
        self.decode = DepthFormerDecode(in_channels=self.encode.list_feats, requires_grad=requires_grad)
        self.list_feats = self.encode.list_feats

    def extract_feats(self, x):
        return self.decode.extract_feats(self.encode(x))

    def forward(self, x):
        feats = self.encode(x)
        return self.decode(feats)
