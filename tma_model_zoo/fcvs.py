from lib2to3.pgen2.tokenize import TokenError
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

from .gru import GRUUNet
from .basics.geometry import tensor_warping
from .basics.activation import stable_softmax
from .basics.dynamic_conv import Deconv, DynamicConv2d
from .basics.upsampling import Upsample
from .universal import VGGUNet, ResnetUNet
from .depth_volumes import DepthVolume1D, BaseDepthVolumeModel


class FCVS(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=False):
        super().__init__(depth_start, depth_end, depth_num, memory_saving)
        self.memory_saving = memory_saving

        reduce_feats = 16
        merge_feats = 64
        act = nn.LeakyReLU(inplace=True)

        self.enc_net = ResnetUNet(n_encoder_stages=3)
        self.merge_net = GRUUNet(64 + 4 + 6, enc_channels=[64, 128, 256, 512], dec_channels=[256, 128, 64], n_enc_convs=3, n_dec_convs=3, act=act)
        self.reduce_conv = DynamicConv2d(64, reduce_feats, stride=2)

        self.mask_feat_conv = nn.Conv2d(68 * 4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_conv = nn.Conv2d(32 + 1, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.rgb_conv = nn.Conv2d(merge_feats, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha_conv = nn.Conv2d(merge_feats, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.depth_volume_model = DepthVolume1D(depth_start=depth_start, depth_end=depth_end, depth_num=depth_num, n_feats=reduce_feats)
        self.upsample = Upsample(mode='bilinear')

    def warp_images(self, src_images, sampling_maps, n_views):
        warped_imgs_srcs = []
        for d in range(self.depth_num):
            feature_list = []
            for view in range(n_views):
                sampling_map = sampling_maps[:, view, d, :, :, :]
                warped_view_feature = tensor_warping(src_images[:, view, ...], sampling_map)
                feature_list.append(warped_view_feature)

            src_features = torch.stack(feature_list, dim=0)  # src_features: [V, N, C, H, W]
            warped_imgs_srcs.append(src_features)
        warped_imgs_srcs = torch.stack(warped_imgs_srcs, dim=1)  # [N, D, 1, H, W]
        return warped_imgs_srcs

    def forward(self, src_colors, prj_depths, sampling_maps, ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, n_channels, height, width = src_colors.shape

        valid_masks = (prj_depths > 0).float()
        valid_mask = (torch.sum(valid_masks, dim=1) > 0).float()

        src_colors = src_colors.view(n_samples * n_views, n_channels, height, width)
        src_feats = self.enc_net(src_colors)
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

        return (fused, deep_colors, warped_imgs_srcs), valid_mask

    def reshape_feats(self, src_colors, valid_mask, n_samples, n_views, n_channels, height, width, src_feats, prj_feats):
        reduced_feats = self.reduce_conv(src_feats)

        mask_feats = self.mask_feat_conv(prj_feats.view(n_samples, -1, height, width))
        valid_mask = nn.functional.interpolate(valid_mask, size=(mask_feats.shape[2], mask_feats.shape[3]), mode='nearest')
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
        prj_feats = functional.grid_sample(src_feats, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        prj_feats = prj_feats.view(n_samples, n_views, -1, height, width)

        prj_colors = functional.grid_sample(src_colors, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
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
        
    def ray_rendering(self, src_colors, reduced_feats, positions, iheight, iwidth,
                      ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, _, height, width = reduced_feats.shape
        _depth_probs, _src_weights = self.depth_volume_model(reduced_feats, ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics, height, width, positions)

        depth_probs = torch.zeros((n_samples, self.depth_num, 1, height, width), device=src_colors.device, dtype=_depth_probs.dtype)
        src_weights = torch.zeros((n_samples, n_views, self.depth_num, height, width), device=src_colors.device, dtype=_depth_probs.dtype)

        for i in range(n_samples):
            depth_probs[i, :, :, positions[i, :, 1], positions[i, :, 2]] = _depth_probs[i, ..., 0, :]
            src_weights[i, :, :, positions[i, :, 1], positions[i, :, 2]] = _src_weights[i, ..., 0, :]

        src_weights = src_weights.reshape(n_samples * n_views * self.depth_num, 1, height, width)
        depth_probs = depth_probs.reshape(n_samples * self.depth_num, 1, height, width)
        src_weights = self.upsample(src_weights, size=(iheight, iwidth)).view(n_samples, n_views, self.depth_num, iheight, iwidth)
        depth_probs = self.upsample(depth_probs, size=(iheight, iwidth)).view(n_samples, self.depth_num, 1, iheight, iwidth)
        
        src_weights_softmax = stable_softmax(src_weights, dim=1)
        depth_prob_volume_softmax = stable_softmax(depth_probs, dim=1)

        # =============================== warp images =========================================
        sampling_maps, src_masks = self.compute_sampling_maps(n_samples, n_views, ys_dst, xs_dst, ys_src, xs_src, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)
        src_masks = src_masks.view(n_samples, n_views, -1, iheight, iwidth)

        warped_imgs_srcs = self.warp_images(src_colors, sampling_maps, n_views).permute(2, 0, 1, 3, 4, 5)  # [D, N, B, H, W, 3], # [D, N, B, H, W, 1]

        # =============== handle source weights with masks (valid warp pixels) ===========
        src_weights_softmax = src_weights_softmax * src_masks  # [N, V, D, H, W, 1]
        src_weights_softmax_sum = torch.sum(src_weights_softmax, dim=1, keepdims=True)
        src_weights_softmax_sum_zero_add = (src_weights_softmax_sum == 0.0).float() * 1e-7
        src_weights_softmax_sum += src_weights_softmax_sum_zero_add
        src_weights_softmax = src_weights_softmax / src_weights_softmax_sum

        # =============== Compute aggregated images =====================================
        weighted_src_img = torch.sum(src_weights_softmax.view(n_samples, n_views, self.depth_num, 1, iheight, iwidth) * warped_imgs_srcs, dim=1) # [D, B, H, W, 3]
        aggregated_img = torch.sum(weighted_src_img * depth_prob_volume_softmax, dim=1)
        warped_imgs_srcs = torch.sum(warped_imgs_srcs * depth_prob_volume_softmax.view(n_samples, 1, self.depth_num, 1, iheight, iwidth), dim=2)

        return aggregated_img, warped_imgs_srcs
