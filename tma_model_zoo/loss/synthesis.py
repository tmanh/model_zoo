import itertools
from sklearn.metrics import nan_euclidean_distances
import torch
import torch.nn as nn
import torch.nn.functional as functional

from tma_model_zoo.utils.misc import tensor2image

from ..basics.geometry import create_sampling_map_src2tgt


class ProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        deep_dst_color, deep_prj_colors = tensors['deep_dst_color'], tensors['deep_prj_colors']
        deep_src_depths, deep_prj_depths = tensors['deep_src_depths'], tensors['deep_prj_depths']
        dst_color, dst_depth, src_depths = tensors['dst_color'], tensors['dst_depth'], tensors['src_depths']
        src_intrinsics, src_extrinsics = tensors['src_intrinsics'], tensors['src_extrinsics']
        dst_intrinsic, dst_extrinsic = tensors['dst_intrinsic'], tensors['dst_extrinsic']

        # get demensions of the tensors
        n_samples, n_views, _, height, width = deep_prj_depths.shape

        dst_depth_dif_loss = 0
        consistency_loss = 0

        # compute the differences between the synthesized color and the color image of the target viewpoint
        tv_loss = self.total_variation_loss(deep_dst_color)
        dst_color_dif_loss = torch.mean(torch.abs(deep_dst_color - dst_color))

        """
        tv_loss += self.total_variation_loss(deep_src_depths.view(n_samples * n_views, -1, height, width))

        # compute the differences between the projected depths and the depth map of the target viewpoint
        for n, i in itertools.product(range(n_samples), range(n_views)):
            sampling_map = create_sampling_map_src2tgt(dst_depth[n], height, width, src_intrinsics[n, i], src_extrinsics[n, i], dst_intrinsic[n, 0], dst_extrinsic[n, 0])
            sampling_map = sampling_map.view(1, 2, height, width).permute(0, 2, 3, 1)
            projected_depth = functional.grid_sample(dst_depth[n].view(1, 1, height, width), sampling_map, mode='bilinear', padding_mode='zeros', align_corners=True)
            dst_depth_dif_loss += torch.mean(torch.abs(deep_src_depths[n, i] - projected_depth) * (projected_depth > 0).float())

        dst_depth_dif_loss /= n_views
        dst_depth_dif_loss += torch.mean(torch.abs(deep_prj_depths - dst_depth.view(n_samples, 1, 1, height, width)))

        # compute the consistency of the depth across multiple views
        for n in range(n_samples):
            for i, j in itertools.product(range(n_views), range(n_views)):
                if j == i:
                    continue

                sampling_map = create_sampling_map_src2tgt(src_depths[n, j], height, width, src_intrinsics[n, i], src_extrinsics[n, i], src_intrinsics[n, j], src_extrinsics[n, j])
                sampling_map = sampling_map.view(1, 2, height, width).permute(0, 2, 3, 1)
                projected_depth = functional.grid_sample(src_depths[n, j].view(1, -1, height, width), sampling_map, mode='bilinear', padding_mode='zeros', align_corners=True)
                consistency_loss += torch.mean(torch.abs(deep_src_depths[n, i] - projected_depth) * (projected_depth > 0).float())
        consistency_loss /= n_views
        """

        return dst_depth_dif_loss + dst_color_dif_loss + consistency_loss + tv_loss * 0.01

    @staticmethod
    def total_variation_loss(x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv


class SynthesisLoss(nn.Module):
    def __init__(self, mode):
        super().__init__()

        self.mode = mode
        self.eps = 1e-7

    def forward(self, refine, deep_dst_color, deep_prj_colors, prj_colors, dst_color):
        tv_loss = 0
        valid_loss = 0

        if refine is not None:
            _valid_loss, _tv_loss = self.compute_single_view_loss(refine, dst_color)
            valid_loss += _valid_loss
            tv_loss += _tv_loss

        if deep_dst_color is not None:
            _valid_loss, _tv_loss = self.compute_single_view_loss(deep_dst_color, dst_color)
            valid_loss += _valid_loss
            tv_loss += _tv_loss

        if deep_prj_colors is not None:
            _valid_loss, _tv_loss = self.compute_multiple_view_loss(deep_prj_colors, dst_color)
            valid_loss += _valid_loss
            tv_loss += _tv_loss

        if prj_colors is not None:
            _valid_loss, _tv_loss = self.compute_multiple_view_loss(prj_colors, dst_color)
            valid_loss += _valid_loss
            tv_loss += _tv_loss

        return valid_loss * 1.0 + tv_loss * 0.01

    def compute_single_view_loss(self, output, target):
        valid_loss = torch.abs(output - target).mean()
        tv_loss = self.total_variation_loss(output)

        return valid_loss, tv_loss

    def compute_multiple_view_loss(self, output, target):
        valid_loss = 0
        tv_loss = 0

        n_views = output.shape[1]
        for i in range(n_views):
            _valid_loss, _tv_loss = self.compute_single_view_loss(output[:, i], target)
            valid_loss += _valid_loss
            tv_loss += _tv_loss
        valid_loss /= n_views
        tv_loss /= n_views

        return valid_loss, tv_loss

    @staticmethod
    def total_variation_loss(x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
