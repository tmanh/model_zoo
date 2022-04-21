# https://ieeexplore.ieee.org/document/9191159
# https://www.mdpi.com/1424-8220/21/14/4892
import time
import torch
import torch.nn as nn

from .depth_completion_guided_unet import GuidedEfficientNet
from ..universal.resnet import Resnet
from ..basics.mapnet import MapNet, ResidualMapNet


class LightDMSR(nn.Module):
    def __init__(self, n_feats=64, n_resblock=4):
        super().__init__()

        self.upscale = ResidualMapNet
        self.depth_completion = GuidedEfficientNet(n_feats, act=nn.LeakyReLU(inplace=True), mode='efficient-rgbm-residual')
        self.refine = Resnet(in_dim=n_feats, n_feats=n_feats, kernel_size=3, n_resblock=n_resblock, out_dim=1, tail=True)

    @staticmethod
    def backbone_out_size(in_h, in_w):
        return in_h, in_w

    def norm_input_data(self, depth_lr, color_lr):
        norm_depth = depth_lr * 2 - 1
        norm_color = color_lr * 2 - 1
        return norm_depth, norm_color

    def extract_features(self, norm_depth, norm_color):
        depth_feature = self.d_1(norm_depth)
        color_feature = self.c_1(norm_color)
        return depth_feature, color_feature

    def extract_upscaled_features(self, color_feature, sr_coarse):
        sr_feature = self.d_1(sr_coarse)
        return torch.cat([sr_feature, color_feature], dim=1)

    def forward(self, depth_lr, color_lr, mask_lr, pos_mat, mapping_mat, mask=None):
        depth_ilr, depth_feats = self.depth_completion(color_lr, depth_lr, mask_lr)

        sr_coarse = self.upscale(pos_mat, mapping_mat, depth_feats, depth_ilr)
        sr_refine = sr_coarse + self.refine(sr_coarse)
       
        return sr_refine, sr_coarse


class BaseDMSR(nn.Module):
    def __init__(self, n_feats, n_resblock, missing, mapnet):
        super().__init__()

        self.missing = missing

        in_dim = 2 if self.missing else 1

        self.d_1 = Resnet(in_dim=in_dim, n_feats=n_feats, kernel_size=3, n_resblock=n_resblock, out_dim=n_feats, tail=True)
        self.d_2 = Resnet(in_dim=n_feats, n_feats=n_feats, kernel_size=3, n_resblock=n_resblock, out_dim=n_feats, tail=True)
        self.c_1 = Resnet(in_dim=3, n_feats=n_feats, kernel_size=3, n_resblock=n_resblock, out_dim=n_feats, tail=True)
        self.refine = Resnet(in_dim=n_feats * 2, n_feats=n_feats, kernel_size=3, n_resblock=n_resblock, out_dim=1, tail=True)

        self.upscale = mapnet

    @staticmethod
    def backbone_out_size(in_h, in_w):
        return in_h, in_w

    def forward(self, depth_lr, color_hr, pos_mat, mapping_mat, mask=None):
        ttt = time.time()

        norm_depth, norm_color = self.norm_input_data(depth_lr, color_hr)
        depth_feature, color_feature = self.extract_features(norm_depth, norm_color, mask)

        sr_coarse = self.upscale(pos_mat, mapping_mat, depth_feature, norm_depth)

        merged_features = self.extract_upscaled_features(color_feature, sr_coarse)
        sr_coarse, sr_refine = self.compute_sr_depth_map(sr_coarse, merged_features)

        elapsed = time.time() - ttt
        return sr_refine, None, [sr_coarse], elapsed

    def compute_sr_depth_map(self, sr_coarse, merged_features):
        sr_residual = self.refine(merged_features)
        sr_coarse, sr_refine = self.denorm_output_data(sr_coarse, sr_residual)
        return sr_coarse, sr_refine

    def extract_upscaled_features(self, color_feature, sr_coarse):
        sr_feature = self.d_1(sr_coarse)
        return torch.cat([sr_feature, color_feature], dim=1)

    def extract_features(self, norm_depth, norm_color, mask):
        depth_feature_1 = self.d_1(torch.cat([norm_depth, mask], dim=1)) if self.missing else self.d_1(norm_depth)
        depth_feature_2 = self.d_2(depth_feature_1)
        color_feature = self.c_1(norm_color)
        return depth_feature_2, color_feature

    def denorm_output_data(self, sr_coarse, sr_residual):
        sr_refine = (sr_coarse + sr_residual + 1) / 2
        sr_coarse = (sr_coarse + 1) / 2
        return sr_coarse, sr_refine

    def norm_input_data(self, depth_lr, color_hr):
        norm_depth = depth_lr * 2 - 1
        norm_color = color_hr * 2 - 1
        return norm_depth, norm_color


class DMSR(BaseDMSR):
    def __init__(self, n_feats, n_resblock, missing):
        super().__init__(n_feats, n_resblock, missing, MapNet(in_channels=n_feats))


class ResidualDMSR(BaseDMSR):
    def __init__(self, n_feats, n_resblock, missing):
        super().__init__(n_feats, n_resblock, missing, ResidualMapNet(in_channels=n_feats * 4, device=self.device))
