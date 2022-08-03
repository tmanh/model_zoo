# https://ieeexplore.ieee.org/document/9191159
# https://www.mdpi.com/1424-8220/21/14/4892
import time
import math
import torch
import torch.nn as nn

from .depth_completion_guided_unet import GuidedEfficientNet
from ..universal.resnet import Resnet
from ..basics.mapnet import *


def default_conv(in_channels, out_channels, kernel_size, bias=True, padding=-1, dilation=1, stride=1):
    if padding == -1:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias,
                         dilation=dilation, stride=stride)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias,
                         dilation=dilation, stride=stride)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, act=nn.LeakyReLU(inplace=True)):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class LightDMSR(nn.Module):
    def __init__(self, n_feats=64, n_resblock=4):
        super().__init__()

        self.upscale = ResidualMapNet
        self.depth_completion = GuidedEfficientNet(n_feats, act==nn.LeakyReLU(inplace=True), mode='efficient-rgbm-residual')
        self.refine = Resnet(in_dim=n_feats, n_feats=n_feats, kernel_size=3, n_resblock=n_resblock, out_dim=1, tail=True)

    @staticmethod
    def backbone_out_size(in_h, in_w):
        return in_h, in_w

    def extract_features(self, depth, color):
        depth_feature = self.d_1(depth)
        color_feature = self.c_1(color)
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
    def __init__(self, n_feats, n_resblock, missing, mapnet, device='cuda'):
        super().__init__()

        self.init_model(n_feats, n_resblock, missing, mapnet, device)

    def init_model(self, n_feats, n_resblock, missing, mapnet, device):
        self.missing = missing

        in_dim = 2 if self.missing else 1
        self.device = device

        # feature extraction
        m_body = [ResBlock(default_conv, n_feats, 3, act=nn.LeakyReLU(inplace=True)) for _ in range(n_resblock)]
        m_body.append(default_conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*m_body)
        self.head = nn.Sequential(*[default_conv(in_dim, n_feats, 3)])

        # generate depth maps of virtual viewpoints
        m_body_2 = [ResBlock(default_conv, n_feats * 4, 3, act=nn.LeakyReLU(inplace=True)) for _ in range(n_resblock)]
        m_body_2.append(default_conv(n_feats * 4, n_feats * 4, 3))
        self.body_up = nn.Sequential(*m_body_2)
        self.head_up = nn.Sequential(*[default_conv(n_feats, n_feats * 4, 3)])

        # extract deep rgb features
        m_body_3 = [ResBlock(default_conv, n_feats, 3, act=nn.LeakyReLU(inplace=True)) for _ in range(n_resblock)]
        m_body_3.append(default_conv(n_feats, n_feats, 3))
        self.body_rgb = nn.Sequential(*m_body_3)
        self.head_rgb = nn.Sequential(*[default_conv(3, n_feats, 3)])

        # compute residual of the color for the refinement
        m_body_4 = [ResBlock(default_conv, n_feats, 3, act=nn.LeakyReLU(inplace=True)) for _ in range(n_resblock)]
        m_body_4.append(default_conv(n_feats, 1, 3))
        self.body_end = nn.Sequential(*m_body_4)
        self.head_end = nn.Sequential(*[default_conv(n_feats * 4 + 1, n_feats, 3)])

        self.upscale = mapnet

    @staticmethod
    def generate_sr_map(in_h, in_w, scale=None, out_h=None, out_w=None, add_scale=True, device='cuda'):
        """
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        """
        scale, out_h, out_w = BaseDMSR.determine_output_shape(in_h, in_w, scale, out_h, out_w)
        scale_mat = BaseDMSR.init_scale_matrix(scale, out_h, out_w, add_scale, device)

        # The idea here is to compute the relative location of the interpolation like in the traditional upsampling methods
        # projection coordinate and calculate the offset
        offset_h = BaseDMSR.generate_coordinate_vector(scale, in_h, out_h, device=device)
        offset_w = BaseDMSR.generate_coordinate_vector(scale, in_w, out_w, device=device)

        # the size is scale_int* inH* (scale_int*inW)
        offset_h_coord = offset_h.repeat(1, out_w).contiguous().view((-1, out_w, 1))
        offset_w_coord = offset_w.repeat(1, out_h).contiguous().view((-1, out_h, 1)).permute((1, 0, 2))

        relative_offset_mat = torch.cat((offset_h_coord, offset_w_coord), dim=2)

        if add_scale:
            relative_offset_mat = torch.cat((scale_mat.view((out_h, out_w, 1)), relative_offset_mat), dim=2)

        # out_h * out_w * 2, out_h = scale_int * inH, out_w = scale_int * inW
        return relative_offset_mat

    @staticmethod
    def generate_coordinate_vector(scale, in_length, out_length, device):
        project_coordinates = torch.arange(0, out_length, 1, device=device).float() / scale
        int_project_coordinates = torch.round(project_coordinates)
        int_project_coordinates[int_project_coordinates >= in_length] = in_length - 1 
        return (project_coordinates - int_project_coordinates).view(out_length, 1).contiguous()

    @staticmethod
    def determine_output_shape(in_h, in_w, scale, out_h, out_w):
        assert scale is not None or (out_h is not None and out_w is not None), "Scale or output shape need to be defined."

        if out_w is None or out_h is None:
            out_h, out_w = int(math.floor(scale * in_h)), int(math.floor(scale * in_w))
        
        scale = out_h / in_h

        return scale, out_h, out_w

    @staticmethod
    def init_scale_matrix(scale, out_h, out_w, add_scale, device):
        scale_mat = None
        if add_scale:
            scale_mat = torch.zeros((1, 1), device=device)
            scale_mat[0, 0] = 1.0 / scale
            scale_mat = torch.cat([scale_mat] * (out_h * out_w), 0)
        return scale_mat

    @staticmethod
    def backbone_out_size(in_h, in_w):
        return in_h, in_w

    def forward(self, depth_lr, depth_bicubic, color_hr, mask_lr, pos_mat=None, mapping_mat=None):
        if mapping_mat is None or pos_mat is None:
            pos_mat = BaseDMSR.generate_sr_map(in_h=depth_lr.shape[-2], in_w=depth_lr.shape[-1], out_h=depth_bicubic.shape[-2], out_w=depth_bicubic.shape[-1], device=color_hr.device)
            pos_mat = pos_mat.unsqueeze(0)

        ttt = time.time()

        if self.missing:
            shallow_features = self.head(torch.cat([depth_lr, mask_lr], dim=1))
        else:
            shallow_features = self.head(depth_lr)

        deep_features = self.body(shallow_features)
        deep_features += shallow_features

        deep_features_2 = self.head_up(deep_features)
        deep_features_2 = self.body_up(deep_features_2)

        coarse, feats = self.upscale(pos_mat, deep_features_2, depth_lr, depth_bicubic, intermediate=True)

        """
        rgb_shallow_features = self.head_rgb(color_hr[:, :, :depth_bicubic.shape[-2], :depth_bicubic.shape[-1]])
        rgb_deep_features = self.body_rgb(rgb_shallow_features)
        rgb_deep_features += rgb_shallow_features
        """

        residual = self.head_end(torch.cat([coarse, feats], dim=1))
        residual = self.body_end(residual)
        refine = coarse + residual

        elapsed = time.time() - ttt
        return refine, [coarse], elapsed


class DMSR(BaseDMSR):
    def __init__(self, n_feats, n_resblock, missing, device='cuda'):
        super().__init__(n_feats, n_resblock, missing, MapNet(in_channels=n_feats), device=device)


class ResidualDMSR(BaseDMSR):
    def __init__(self, n_feats, n_resblock, missing, device='cuda'):
        super().__init__(n_feats, n_resblock, missing, ResidualMapNet(in_channels=n_feats * 4), device=device)


class TopDMSR(BaseDMSR):
    def __init__(self, in_feats, n_feats, n_resblock, missing, device='cuda'):
        self.in_feats = in_feats
        super().__init__(n_feats, n_resblock, missing, ResidualMapNet(in_channels=n_feats * 4), device=device)

    def init_model(self, n_feats, n_resblock, missing, mapnet, device):
        self.missing = missing

        in_dim = self.in_feats + 1 if self.missing else self.in_feats
        self.device = device

        # feature extraction
        m_body = [ResBlock(default_conv, n_feats, 3, act=nn.LeakyReLU(inplace=True)) for _ in range(n_resblock)]
        m_body.append(default_conv(n_feats, n_feats * 4, 3))
        self.body = nn.Sequential(*m_body)
        self.head = nn.Sequential(*[default_conv(in_dim, n_feats, 3)])

        # compute residual of the color for the refinement
        m_body_4 = [ResBlock(default_conv, n_feats, 3, act=nn.LeakyReLU(inplace=True)) for _ in range(n_resblock)]
        m_body_4.append(default_conv(n_feats, 1, 3))
        self.body_end = nn.Sequential(*m_body_4)
        self.head_end = nn.Sequential(*[default_conv(n_feats * 4 + 1, n_feats, 3)])

        self.upscale = mapnet
    
    def forward(self, feats, depth_lr, depth_bicubic, color_hr, mask_lr, pos_mat=None, mapping_mat=None):
        if mapping_mat is None or pos_mat is None:
            pos_mat = BaseDMSR.generate_sr_map(in_h=feats.shape[-2], in_w=feats.shape[-1], out_h=depth_bicubic.shape[-2], out_w=depth_bicubic.shape[-1], device=color_hr.device)
            pos_mat = pos_mat.unsqueeze(0)

        ttt = time.time()

        if self.missing:
            shallow_features = self.head(torch.cat([feats, mask_lr], dim=1))
        else:
            shallow_features = self.head(feats)

        deep_features = self.body(shallow_features)

        coarse, feats = self.upscale(pos_mat, deep_features, depth_lr, depth_bicubic, intermediate=True)

        residual = self.head_end(torch.cat([coarse, feats], dim=1))
        residual = self.body_end(residual)
        refine = coarse + residual

        elapsed = time.time() - ttt
        return refine, [coarse], elapsed


class ResidualDMSR2(BaseDMSR):
    def init_model(self, n_feats, n_resblock, missing, mapnet, device):
        self.missing = missing

        in_dim = 2 if self.missing else 1
        self.device = device

        # feature extraction
        self.shallow = self.create_resnet(in_dim, out_dim=n_feats, n_feats=n_feats, n_resblock=n_resblock // 2)

        # deep feats
        self.deep_1 = self.create_resnet(n_feats, out_dim=n_feats, n_feats=n_feats, n_resblock=n_resblock // 2)
        self.deep_2 = self.create_resnet(n_feats, out_dim=n_feats, n_feats=n_feats, n_resblock=n_resblock // 2)
        self.deep_3 = self.create_resnet(n_feats, out_dim=n_feats, n_feats=n_feats, n_resblock=n_resblock // 2)
        self.deep_4 = self.create_resnet(n_feats, out_dim=n_feats, n_feats=n_feats, n_resblock=n_resblock // 2)

        # compute residual of the color for the refinement
        self.refine = self.create_resnet((n_feats + 1), out_dim=1, n_feats=n_feats, n_resblock=n_resblock // 2)
        self.fuse = self.create_resnet(4, out_dim=4, n_feats=n_feats, n_resblock=n_resblock // 4)

        self.upscale_1 = mapnet(n_feats)
        self.upscale_2 = mapnet(n_feats)
        self.upscale_3 = mapnet(n_feats)
        self.upscale_4 = mapnet(n_feats)

    def create_resnet(self, in_dim, out_dim, n_feats, n_resblock):
        resnet = [default_conv(in_dim, n_feats, 3)]
        resnet.extend([ResBlock(default_conv, n_feats, 3, act=nn.LeakyReLU(inplace=True)) for _ in range(n_resblock)])
        resnet.append(default_conv(n_feats, out_dim, 3))
        return nn.Sequential(*resnet)

    def forward(self, depth_lr, depth_bicubic, color_hr, pos_mat=None, mapping_mat=None, mask_lr=None):
        if mapping_mat is None or pos_mat is None:
            pos_mat = BaseDMSR.generate_sr_map(in_h=depth_lr.shape[-2], in_w=depth_lr.shape[-1], out_h=depth_bicubic.shape[-2], out_w=depth_bicubic.shape[-1], device=color_hr.device)
            pos_mat = pos_mat.unsqueeze(0)

        ttt = time.time()
        shallow = self.shallow(torch.cat([depth_lr, mask_lr], dim=1)) if self.missing else self.shallow(depth_lr)

        deep_1 = self.deep_1(shallow) + shallow
        deep_2 = self.deep_2(shallow) + shallow
        deep_3 = self.deep_3(shallow) + shallow
        deep_4 = self.deep_4(shallow) + shallow

        coarse_1, cfeat_1 = self.upscale_1(pos_mat, deep_1, depth_lr, depth_bicubic, intermediate=True)
        coarse_2, cfeat_2 = self.upscale_2(pos_mat, deep_2, depth_lr, depth_bicubic, intermediate=True)
        coarse_3, cfeat_3 = self.upscale_3(pos_mat, deep_3, depth_lr, depth_bicubic, intermediate=True)
        coarse_4, cfeat_4 = self.upscale_4(pos_mat, deep_4, depth_lr, depth_bicubic, intermediate=True)

        refine_1 = coarse_1 + self.refine(torch.cat([coarse_1, cfeat_1], dim=1))
        refine_2 = coarse_2 + self.refine(torch.cat([coarse_2, cfeat_2], dim=1))
        refine_3 = coarse_3 + self.refine(torch.cat([coarse_3, cfeat_3], dim=1))
        refine_4 = coarse_4 + self.refine(torch.cat([coarse_4, cfeat_4], dim=1))

        refine = torch.cat([refine_1, refine_2, refine_3, refine_4], dim=1)
        scores = torch.softmax(self.fuse(refine), dim=1)

        final = torch.sum(refine * scores, dim=1, keepdims=True)

        elapsed = time.time() - ttt
        return final, [coarse_1, coarse_2, coarse_3, coarse_4, refine_1, refine_2, refine_3, refine_4], elapsed
