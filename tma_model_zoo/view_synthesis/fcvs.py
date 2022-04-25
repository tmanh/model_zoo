import itertools
import math
from kornia import depth
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..universal import *
from ..basics.geometry import tensor_warping
from ..basics.activation import stable_softmax
from ..basics.dynamic_conv import Deconv, DynamicConv2d
from ..basics.upsampling import Upsample
from ..depth_volumes import DepthVolume1D, BaseDepthVolumeModel
from ..basics.mapnet import ResidualPlusMapNet


class FCVS(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=False):
        super().__init__(depth_start, depth_end, depth_num, memory_saving)
        self.memory_saving = memory_saving

        reduce_feats = 16
        merge_feats = 64
        act = nn.LeakyReLU(inplace=True)

        # source feats
        self.enc_net = SNetDS2BN_base_8(in_channels=3)

        # depth volume and ray rendering
        self.depth_volume_model = DepthVolume1D(depth_start=depth_start, depth_end=depth_end, depth_num=depth_num, n_feats=reduce_feats)

        # sr prob maps
        self.prob_feat_conv = Resnet(in_dim=self.depth_num, n_feats=64, kernel_size=3, n_resblock=4, out_dim=64, tail=True)
        self.weight_feat_conv = Resnet(in_dim=self.depth_num, n_feats=64, kernel_size=3, n_resblock=4, out_dim=64, tail=True)
        self.upscale_probs = ResidualPlusMapNet(in_channels=4, n_feats=64, out_channels=48)
        self.upscale_weights = ResidualPlusMapNet(in_channels=4, n_feats=64, out_channels=48)

        self.freeze_visibility()

        self.merge_net = GRUUNet(64 + 4 + 6, enc_channels=[64, 128, 256, 512], dec_channels=[256, 128, 64], n_enc_convs=3, n_dec_convs=3, act=act)
        self.reduce_conv = DynamicConv2d(64, reduce_feats, stride=1)

        self.mask_feat_conv = nn.Conv2d(68 * 4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_conv = nn.Conv2d(32 + 1, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.rgb_conv = nn.Conv2d(merge_feats, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha_conv = nn.Conv2d(merge_feats, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.sheight = 128
        self.swidth = 160

    def freeze_visibility(self):
        for param in self.enc_net.parameters():
            param.requires_grad = False
        for module in self.enc_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.momentum = 0
                module.eval()
        for param in self.depth_volume_model.parameters():
            param.requires_grad = False
        for module in self.enc_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.momentum = 0
                module.eval()

    def forward(self, src_colors, prj_depths, sampling_maps, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, n_channels, height, width = src_colors.shape

        src_colors = src_colors.view(-1, n_channels, height, width)
        src_feats = self.enc_net(src_colors)

        reduced_feats = functional.interpolate(src_feats, size=(self.sheight, self.swidth), mode='nearest')

        src_colors = src_colors.view(n_samples, n_views, n_channels, height, width)
        reduced_feats = reduced_feats.view(n_samples, n_views, -1, self.sheight, self.swidth)

        positions = None
        aggregated_img, warped_imgs_srcs = self.ray_rendering(src_colors, reduced_feats, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)

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

    def sr_prob(self, depth_probs, view_weights, original_height, original_width):
        n_samples, n_views, n_depths, height, width = view_weights.shape

        pos_mat, mapping_mat = self.generate_sr_map(height, width, original_height, original_width, depth_probs.device)
        pos_mat = pos_mat.view(1, original_height, original_width, pos_mat.shape[2])
        pos_mat = pos_mat.repeat(n_samples, 1, 1, 1)

        reshaped_probs = depth_probs.view(n_samples, n_depths, height, width)
        reshape_weights = view_weights.view(n_samples * n_views, n_depths, height, width)

        prob_feature = self.prob_feat_conv(reshaped_probs)
        weight_feature = self.weight_feat_conv(reshape_weights)

        sr_probs = self.upscale_probs(pos_mat, mapping_mat, prob_feature, reshaped_probs)
        sr_weights = self.upscale_weights(pos_mat, mapping_mat, weight_feature, reshape_weights)

        return sr_probs.view(n_samples, n_depths, 1, original_height, original_width), sr_weights.view(n_samples, n_views, n_depths, original_height, original_width)

    # by given the scale and the size of input image
    # we calculate the input matrix for the weight prediction network
    # input matrix for weight prediction network
    def generate_sr_map(self, in_h, in_w, out_h, out_w, device, add_scale=True):
        scale_h = in_h / out_h
        scale_w = in_w / out_w

        scale_mat_h = None
        scale_mat_w = None

        if add_scale:
            scale_mat_h = torch.zeros((1, 1))
            scale_mat_h[0, 0] = 1.0 / scale_h
            scale_mat_h = torch.cat([scale_mat_h] * (out_h * out_w), 0)

            scale_mat_w = torch.zeros((1, 1))
            scale_mat_w[0, 0] = 1.0 / scale_w
            scale_mat_w = torch.cat([scale_mat_w] * (out_h * out_w), 0)

        # projection coordinate and calculate the offset
        h_project_coord = torch.mul(torch.arange(0, out_h, 1).float(), scale_h)
        int_h_project_coord = torch.floor(h_project_coord)

        w_project_coord = torch.mul(torch.arange(0, out_w, 1).float(), scale_w)
        int_w_project_coord = torch.floor(w_project_coord)

        # set the proper view compute offset
        offset_h = (h_project_coord - int_h_project_coord).view(out_h, 1).contiguous()
        int_h_project_coord = int_h_project_coord.long().view(out_h, 1).contiguous()

        offset_w = (w_project_coord - int_w_project_coord).view(out_w, 1).contiguous()
        int_w_project_coord = int_w_project_coord.long().view(out_w, 1).contiguous()

        # the size is scale_int* inH* (scale_int*inW)
        offset_h_coord = offset_h.repeat(1, out_w).contiguous().view((-1, out_w, 1))
        offset_w_coord = offset_w.repeat(1, out_h).contiguous().view((-1, out_h, 1)).permute((1, 0, 2))

        int_h_project_coord_all = int_h_project_coord.repeat(1, out_w).contiguous().view((out_h, out_w, 1))
        int_w_project_coord_all = int_w_project_coord.repeat(1, out_h).contiguous().view((-1, out_h, 1))
        int_w_project_coord_all = int_w_project_coord_all.permute((1, 0, 2))

        position_mat = torch.cat((offset_h_coord, offset_w_coord), dim=2)
        mapping_mat = torch.cat((int_h_project_coord_all.long(), int_w_project_coord_all.long()), dim=2)

        if add_scale:
            scale_mat_h = scale_mat_h.view((out_h, out_w, 1))
            scale_mat_w = scale_mat_w.view((out_h, out_w, 1))
            position_mat = torch.cat((scale_mat_h, scale_mat_w, position_mat), dim=2)

        # out_h * out_w * 2, out_h = scale_int * inH, out_w = scale_int * inW
        return position_mat.to(device), mapping_mat.to(device)

    def ray_rendering(self, src_colors, reduced_feats, original_height, original_width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        view_weights, depth_probs = self.compute_probs(src_colors, reduced_feats, original_height, original_width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        warped_masks, warped_imgs_srcs = self.warp_images(src_colors, original_height, original_width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)

        # self.save_images(warped_imgs_srcs, depth_prob_volume_softmax, n_samples, iheight, iwidth)
        return self.fuse_rays(warped_masks, warped_imgs_srcs, view_weights, depth_probs, original_height, original_width)

    def fuse_rays(self, warped_masks, warped_imgs_srcs, view_weights, depth_probs, original_height, original_width):
        n_samples, n_views, _, _, _ = warped_masks.shape

        # =============== handle source weights with masks (valid warp pixels) ===========
        view_weights = view_weights * warped_masks  # [N, V, D, H, W, 1]
        src_weights_softmax_sum = torch.sum(view_weights, dim=1, keepdims=True)
        src_weights_softmax_sum_zero_add = (src_weights_softmax_sum == 0.0).float() + 1e-7
        src_weights_softmax_sum += src_weights_softmax_sum_zero_add
        view_weights = view_weights / src_weights_softmax_sum

        # =============== Compute aggregated images =====================================
        weighted_src_img = torch.sum(view_weights.view(n_samples, n_views, self.depth_num, 1, original_height, original_width) * warped_imgs_srcs, dim=1) # [D, B, H, W, 3]
        aggregated_img = torch.sum(weighted_src_img * depth_probs, dim=1)
        warped_imgs_srcs = torch.sum(warped_imgs_srcs * depth_probs.view(n_samples, 1, self.depth_num, 1, original_height, original_width), dim=2)

        return aggregated_img, warped_imgs_srcs

    def compute_probs(self, src_colors, reduced_feats, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, _, height, width = reduced_feats.shape
        coarse_depth_probs, coarse_src_weights = self.coarse_prob_volumes_from(reduced_feats, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics, height, width)
        depth_probs, view_weights = self.extract_subset_prob_volumes(src_colors, n_samples, n_views, height, width, coarse_depth_probs, coarse_src_weights)
        
        depth_probs, view_weights = self.sr_prob(depth_probs, view_weights, iheight, iwidth)
        # depth_probs, view_weights = self.upscale_prob_volumes(iheight, iwidth, n_samples, n_views, height, width, depth_probs, view_weights)

        view_weights_softmax = torch.softmax(view_weights, dim=1)
        depth_prob_volume_softmax = torch.softmax(depth_probs, dim=1)
        return view_weights_softmax, depth_prob_volume_softmax

    def extract_subset_prob_volumes(self, src_colors, n_samples, n_views, height, width, _depth_probs, _view_weights):
        """First, this function will detect the positions which are not so "good" (color values are not consistent, no color values, etc.)
        Second, this function will extract the probability values at the detected locations

        Args:
            src_colors: source images
            positions: the detected locations
            n_samples: the number of samples
            n_views: the number of viewpoints
            height: the height of the images
            width: the width of the images
            _depth_probs: the probability volumes (which depth is the best)
            _view_weights: the probability volumes (which view is the best)

        Returns:
            depth_probs: the probability volumes (which depth is the best)
            view_weights: the probability volumes (which view is the best)
        """
        positions = None

        if positions is not None:
            depth_probs = torch.zeros((n_samples, self.depth_num, 1, height, width), device=src_colors.device, dtype=_depth_probs.dtype)
            view_weights = torch.zeros((n_samples, n_views, self.depth_num, height, width), device=src_colors.device, dtype=_depth_probs.dtype)

            for i in range(n_samples):
                depth_probs[i, :, :, positions[i, :, 1], positions[i, :, 2]] = _depth_probs[i, ..., 0, :]
                view_weights[i, :, :, positions[i, :, 1], positions[i, :, 2]] = _view_weights[i, ..., 0, :]
        else:
            depth_probs = _depth_probs
            view_weights = _view_weights
        return depth_probs, view_weights

    def warp_images(self, src_colors, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, _, _, _ = src_colors.shape
        sampling_maps, src_masks = self.compute_sampling_maps(n_samples, n_views, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)
        src_masks = src_masks.view(n_samples, n_views, -1, iheight, iwidth)

        warped_imgs_srcs = self._warp_images(src_colors, sampling_maps, n_views).permute(2, 0, 1, 3, 4, 5)  # [N, V, D, 3, H, W]
        return src_masks, warped_imgs_srcs

    def _warp_images(self, src_images, sampling_maps, n_views):
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

    def upscale_prob_volumes(self, iheight, iwidth, n_samples, n_views, height, width, depth_probs, view_weights):
        """Upsample the probability volumes

        Args:
            iheight: original height
            iwidth: original width
            n_samples: number of samples
            n_views: number of views
            height: downscaled height
            width: downscaled width
            depth_probs: downscaled probability volumes (depth)
            view_weights: downscaled probability volumes (view)

        Returns:
            scaled_depth_probs, scaled_view_weights: the scaled depth probability (depth and view)
        """
        view_weights = view_weights.reshape(n_samples * n_views * self.depth_num, 1, height, width)
        depth_probs = depth_probs.reshape(n_samples * self.depth_num, 1, height, width)
        scaled_view_weights = functional.interpolate(view_weights, size=(iheight, iwidth), mode='nearest').view(n_samples, n_views, self.depth_num, iheight, iwidth)
        scaled_depth_probs = functional.interpolate(depth_probs, size=(iheight, iwidth), mode='nearest').view(n_samples, self.depth_num, 1, iheight, iwidth)
        return scaled_depth_probs, scaled_view_weights

    def coarse_prob_volumes_from(self, reduced_feats, iheight, iwidth, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics, height, width):
        """This function is used to compute the probability volumes of the scaled images

        Args:
            reduced_feats: rescaled feature maps
            iheight: original height
            iwidth: original width
            dst_intrinsics: intrinsic matrices of the target viewpoints
            dst_extrinsics: extrinsic matrices of the target viewpoints
            src_intrinsics: intrinsic matrices of the source viewpoints
            src_extrinsics: extrinsic matrices of the source viewpoints
            height: new height
            width: new width

        Returns:
            depth_probs: probability volumes (depth)
            view_weights: probability volumes (view)
        """
        new_dst_intrinsics, new_src_intrinsics = self.scale_intrinsic_parameters(iheight, iwidth, dst_intrinsics, src_intrinsics, height, width)
        return self.depth_volume_model(reduced_feats, new_dst_intrinsics, dst_extrinsics, new_src_intrinsics, src_extrinsics)

    def scale_intrinsic_parameters(self, iheight, iwidth, dst_intrinsics, src_intrinsics, height, width):
        """Compute the intrinsic parameters for the scaled images

        Args:
            iheight: original height of the images
            iwidth: original width of the images
            dst_intrinsics: the intrinsics of the target viewpoint
            src_intrinsics: the intrinsics of the source viewpoints
            height: height of the scaled images
            width: width of the scaled images

        Returns:
            new_dst_intrinsics: the new intrinsic matrices of the target viewpoint
            new_src_intrinsics: the new intrinsic matrices of the source viewpoints
        """
        scaled_dst_intrinsics, scaled_src_intrinsics = dst_intrinsics.detach().clone(), src_intrinsics.detach().clone()
        sh, sw = height / iheight, width / iwidth
        scaled_dst_intrinsics[:, :, 0, 0] = scaled_dst_intrinsics[:, :, 0, 0] * sw  # N, V, 4, 4
        scaled_dst_intrinsics[:, :, 1, 1] = scaled_dst_intrinsics[:, :, 1, 1] * sh  # N, V, 4, 4
        scaled_dst_intrinsics[:, :, 0, 2] = scaled_dst_intrinsics[:, :, 0, 2] * sw  # N, V, 4, 4
        scaled_dst_intrinsics[:, :, 1, 2] = scaled_dst_intrinsics[:, :, 1, 2] * sh  # N, V, 4, 4
        scaled_src_intrinsics[:, :, 0, 0] = scaled_src_intrinsics[:, :, 0, 0] * sw  # N, V, 4, 4
        scaled_src_intrinsics[:, :, 1, 1] = scaled_src_intrinsics[:, :, 1, 1] * sh  # N, V, 4, 4
        scaled_src_intrinsics[:, :, 0, 2] = scaled_src_intrinsics[:, :, 0, 2] * sw  # N, V, 4, 4
        scaled_src_intrinsics[:, :, 1, 2] = scaled_src_intrinsics[:, :, 1, 2] * sh  # N, V, 4, 4
        return scaled_dst_intrinsics, scaled_src_intrinsics

    def save_images(self, warped_imgs_srcs, depth_prob_volume_softmax, n_samples, height, width):
        import cv2
        import os
        import numpy as np
        xxx = warped_imgs_srcs * depth_prob_volume_softmax.view(n_samples, 1, self.depth_num, 1, height, width)
        for n, v, d in itertools.product(range(1), range(4), range(48)):
            if not os.path.exists(f'{d}'):
                os.makedirs(f'{d}')
            x = xxx[n, v, d] * 255
            x = x.permute(1, 2, 0).view(height, width, 3).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(f'{d}/{n}-{v}-{d}.png', x)
