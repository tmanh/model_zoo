import itertools
from kornia import tensor_to_image
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..universal import *
from ..basics.geometry import tensor_warping
from ..basics.activation import stable_softmax
from ..basics.dynamic_conv import Deconv, DynamicConv2d
from ..basics.upsampling import Upsample
from ..depth_volumes import DepthVolume1D, BaseDepthVolumeModel


class FCVS(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=False):
        super().__init__(depth_start, depth_end, depth_num, memory_saving)
        self.memory_saving = memory_saving

        reduce_feats = 16
        merge_feats = 64
        act = nn.LeakyReLU(inplace=True)

        self.enc_net = SNetDS2BN_base_8(in_channels=3)
        self.merge_net = GRUUNet(64 + 4 + 6, enc_channels=[64, 128, 256, 512], dec_channels=[256, 128, 64], n_enc_convs=3, n_dec_convs=3, act=act)
        self.reduce_conv = DynamicConv2d(64, reduce_feats, stride=1)

        self.mask_feat_conv = nn.Conv2d(68 * 4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_conv = nn.Conv2d(32 + 1, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.rgb_conv = nn.Conv2d(merge_feats, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha_conv = nn.Conv2d(merge_feats, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.depth_volume_model = DepthVolume1D(depth_start=depth_start, depth_end=depth_end, depth_num=depth_num, n_feats=reduce_feats)
        self.upsample = Upsample(mode='nearest')

        self.sheight = 128
        self.swidth = 160

    def warp_images(self, src_images, sampling_maps, n_views):
        warped_imgs_srcs = []
        for d in range(self.depth_num):
            image_list = []
            for view in range(n_views):
                sampling_map = sampling_maps[:, view, d, :, :, :]
                warped_view_image = tensor_warping(src_images[:, view, ...], sampling_map)
                image_list.append(warped_view_image)

            stacked_images = torch.stack(image_list, dim=0)  # src_features: [V, N, C, H, W]
            warped_imgs_srcs.append(stacked_images)
        return torch.stack(warped_imgs_srcs, dim=1)  # [N, D, 1, H, W]

    def forward(self, src_colors, prj_depths, sampling_maps, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, n_channels, height, width = src_colors.shape

        src_colors = src_colors.view(-1, n_channels, height, width)
        src_feats = self.enc_net(src_colors)

        reduced_feats = functional.interpolate(src_feats, size=(self.sheight, self.swidth), mode='nearest')

        src_colors = src_colors.view(n_samples, n_views, n_channels, height, width)
        reduced_feats = reduced_feats.view(n_samples, n_views, -1, self.sheight, self.swidth)

        positions = None
        aggregated_img, warped_imgs_srcs = self.ray_rendering(src_colors, reduced_feats, positions, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)

        valid_masks = (prj_depths > 0).float()
        valid_mask = (torch.sum(valid_masks, dim=1) > 0).float()

        return {'refine': aggregated_img, 'deep_dst_color': None, 'deep_prj_colors': warped_imgs_srcs, 'prj_colors': None, 'dst_color': None, 'deep_src_depths': None,
                'valid_mask': valid_mask}
        exit()

        valid_masks = (prj_depths > 0).float()
        valid_mask = (torch.sum(valid_masks, dim=1) > 0).float()

        # print('src feats', torch.cuda.memory_allocated(0) / 1024 / 1024)
        prj_feats = self.warp_feats(src_colors, prj_depths, sampling_maps, n_samples, n_views, height, width, src_feats)
        # print('prj feats', torch.cuda.memory_allocated(0) / 1024 / 1024)
        src_colors, reduced_feats, mask, positions = self.reshape_feats(src_colors, valid_mask, n_samples, n_views, n_channels, height, width, src_feats, prj_feats)

        mask = nn.functional.interpolate(mask, size=(height, width), mode='nearest').view(n_samples, 1, 1, height, width)
        threshold_mask = (mask > mask.mean()).float()

        _, warped_imgs_srcs = self.ray_rendering(src_colors, reduced_feats, positions, height, width,
                                                 ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        # print('ray rendering', torch.cuda.memory_allocated(0) / 1024 / 1024)

        warped_imgs_srcs = warped_imgs_srcs * threshold_mask
        merged_warped_imgs_srcs = warped_imgs_srcs * mask + prj_feats[:, :, -4:-1, ...] * (1 - threshold_mask * mask)
        prj_feats = torch.cat([prj_feats, warped_imgs_srcs, merged_warped_imgs_srcs], dim=2)
        fused, deep_colors = self.refine(n_views, prj_feats)
        # print('refinement', torch.cuda.memory_allocated(0) / 1024 / 1024)
        return {'refine': fused, 'deep_dst_color': None, 'deep_prj_colors': deep_colors, 'prj_colors': warped_imgs_srcs, 'dst_color': None, 'valid_mask': valid_mask}

    def reshape_feats(self, src_colors, valid_mask, n_samples, n_views, n_channels, height, width, src_feats, prj_feats):
        reduced_feats = self.reduce_conv(src_feats)

        mask_feats = self.mask_feat_conv(prj_feats.view(n_samples, -1, height, width))
        valid_mask = functional.interpolate(valid_mask, size=(mask_feats.shape[2], mask_feats.shape[3]), mode='nearest')
        conf_mask = torch.sigmoid(self.mask_conv(torch.cat([mask_feats, valid_mask], dim=1)))

        src_colors = src_colors.view(n_samples, n_views, n_channels, height, width)
        reduced_feats = reduced_feats.view(n_samples, n_views, -1, height // 2, width // 2)

        positions = []
        for i in range(n_samples):
            position = (conf_mask[i] > conf_mask[i].mean()).nonzero()
            positions.append(position)

        positions = torch.stack(positions, dim=0).view(n_samples, -1, 3)

        return src_colors, reduced_feats, conf_mask, positions

    def warp_feats(self, src_colors, prj_depths, sampling_maps, n_samples, n_views, height, width, src_feats):
        sampling_maps = sampling_maps.reshape(n_samples * n_views, 2, height, width).permute(0, 2, 3, 1)
        prj_feats = functional.grid_sample(src_feats, sampling_maps, mode='nearest', padding_mode='zeros', align_corners=True)
        prj_feats = prj_feats.view(n_samples, n_views, -1, height, width)

        prj_colors = functional.grid_sample(src_colors, sampling_maps, mode='nearest', padding_mode='zeros', align_corners=True)
        prj_colors = prj_colors.view(n_samples, n_views, 3, height, width)

        prj_feats = torch.cat([prj_feats, prj_colors, prj_depths], dim=2)
        return prj_feats

    def refine(self, n_views, prj_feats):
        hs = None
        out_colors = []
        alphas = []
        for vidx in range(n_views):
            y, hs = self.merge_net(prj_feats[:, vidx], hs)

            out_colors.append(self.rgb_conv(y))
            alphas.append(self.alpha_conv(y))

        return self.merge_deep_images(out_colors, alphas)

    def merge_deep_images(self, colors, alphas):
        colors = torch.stack(colors)
        alphas = torch.softmax(torch.stack(alphas), dim=0)
        fused = (alphas * colors).sum(dim=0)
        return fused, colors.permute(1, 0, 2, 3, 4)
        
    def ray_rendering(self, src_colors, reduced_feats, positions, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, _, height, width = reduced_feats.shape

        new_dst_intrinsics, new_src_intrinsics = dst_intrinsics.detach().clone(), src_intrinsics.detach().clone()
        sh, sw = height / iheight, width / iwidth
        new_dst_intrinsics[:, :, 0, 0] = new_dst_intrinsics[:, :, 0, 0] * sw  # N, V, 4, 4
        new_dst_intrinsics[:, :, 1, 1] = new_dst_intrinsics[:, :, 1, 1] * sh  # N, V, 4, 4
        new_dst_intrinsics[:, :, 0, 2] = new_dst_intrinsics[:, :, 0, 2] * sw  # N, V, 4, 4
        new_dst_intrinsics[:, :, 1, 2] = new_dst_intrinsics[:, :, 1, 2] * sh  # N, V, 4, 4
        new_src_intrinsics[:, :, 0, 0] = new_src_intrinsics[:, :, 0, 0] * sw  # N, V, 4, 4
        new_src_intrinsics[:, :, 1, 1] = new_src_intrinsics[:, :, 1, 1] * sh  # N, V, 4, 4
        new_src_intrinsics[:, :, 0, 2] = new_src_intrinsics[:, :, 0, 2] * sw  # N, V, 4, 4
        new_src_intrinsics[:, :, 1, 2] = new_src_intrinsics[:, :, 1, 2] * sh  # N, V, 4, 4

        _depth_probs, _src_weights = self.depth_volume_model(reduced_feats, new_dst_intrinsics, dst_extrinsics, new_src_intrinsics, src_extrinsics, positions)

        if positions is not None:
            depth_probs = torch.zeros((n_samples, self.depth_num, 1, height, width), device=src_colors.device, dtype=_depth_probs.dtype)
            src_weights = torch.zeros((n_samples, n_views, self.depth_num, height, width), device=src_colors.device, dtype=_depth_probs.dtype)

            for i in range(n_samples):
                depth_probs[i, :, :, positions[i, :, 1], positions[i, :, 2]] = _depth_probs[i, ..., 0, :]
                src_weights[i, :, :, positions[i, :, 1], positions[i, :, 2]] = _src_weights[i, ..., 0, :]
        else:
            depth_probs = _depth_probs
            src_weights = _src_weights

        src_weights = src_weights.reshape(n_samples * n_views * self.depth_num, 1, height, width)
        depth_probs = depth_probs.reshape(n_samples * self.depth_num, 1, height, width)
        src_weights = functional.interpolate(src_weights, size=(iheight, iwidth), mode='nearest').view(n_samples, n_views, self.depth_num, iheight, iwidth)
        depth_probs = functional.interpolate(depth_probs, size=(iheight, iwidth), mode='nearest').view(n_samples, self.depth_num, 1, iheight, iwidth)

        src_weights_softmax = torch.softmax(src_weights, dim=1)
        depth_prob_volume_softmax = torch.softmax(depth_probs, dim=1)

        # =============================== warp images =========================================
        sampling_maps, src_masks = self.compute_sampling_maps(n_samples, n_views, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)
        src_masks = src_masks.view(n_samples, n_views, -1, iheight, iwidth)

        warped_imgs_srcs = self.warp_images(src_colors, sampling_maps, n_views).permute(2, 0, 1, 3, 4, 5)  # [N, V, D, 3, H, W]

        """
        import cv2
        import os
        import numpy as np
        xxx = warped_imgs_srcs * depth_prob_volume_softmax.view(n_samples, 1, self.depth_num, 1, iheight, iwidth)
        for n, v, d in itertools.product(range(1), range(4), range(48)):
            if not os.path.exists(f'{d}'):
                os.makedirs(f'{d}')
            x = xxx[n, v, d] * 255
            x = x.permute(1, 2, 0).view(iheight, iwidth, 3).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(f'{d}/{n}-{v}-{d}.png', x)
        # """

        # =============== handle source weights with masks (valid warp pixels) ===========
        src_weights_softmax = src_weights_softmax * src_masks  # [N, V, D, H, W, 1]
        src_weights_softmax_sum = torch.sum(src_weights_softmax, dim=1, keepdims=True)
        src_weights_softmax_sum_zero_add = (src_weights_softmax_sum == 0.0).float() + 1e-7
        src_weights_softmax_sum += src_weights_softmax_sum_zero_add
        src_weights_softmax = src_weights_softmax / src_weights_softmax_sum

        # =============== Compute aggregated images =====================================
        weighted_src_img = torch.sum(src_weights_softmax.view(n_samples, n_views, self.depth_num, 1, iheight, iwidth) * warped_imgs_srcs, dim=1) # [D, B, H, W, 3]
        aggregated_img = torch.sum(weighted_src_img * depth_prob_volume_softmax, dim=1)
        warped_imgs_srcs = torch.sum(warped_imgs_srcs * depth_prob_volume_softmax.view(n_samples, 1, self.depth_num, 1, iheight, iwidth), dim=2)

        return aggregated_img, warped_imgs_srcs
