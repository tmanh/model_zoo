import numpy as np

import torch
import torch.nn as nn

from .universal import *
from .basics.conv_gru import ConvGRU2d
from .basics.padding import same_padding
from .basics.geometry import create_sampling_map_target2source, tensor_warping
from .basics.dynamic_conv import DeconvGroupNorm, DynamicConv2d


class BaseDepthVolumeModel(nn.Module):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=True, inv_depth=False):
        super().__init__()

        self.memory_saving = memory_saving
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.depth_num = depth_num

        if inv_depth:
            self.depth_ranges = 1.0 / np.linspace(1 / self.depth_start, 1 / self.depth_end, self.depth_num)
        else:
            self.depth_ranges = np.linspace(self.depth_start, self.depth_end, self.depth_num)

    def compute_sampling_maps(self, n_samples, n_views, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        sampling_maps = []
        masks = []
        for n in range(n_samples):
            sample_sampling_maps = []
            sample_masks = []
            for i in range(n_views):
                view_sampling_maps = []
                view_masks = []

                for d in self.depth_ranges:
                    dst_intrinsic, dst_extrinsic = dst_intrinsics[n, 0, ...], dst_extrinsics[n, 0, ...]
                    src_intrinsic, src_extrinsic = src_intrinsics[n, i, ...], src_extrinsics[n, i, ...]
                    sampling_map, mask = create_sampling_map_target2source(d, height, width, dst_intrinsic, dst_extrinsic, src_intrinsic, src_extrinsic)
                    view_sampling_maps.append(sampling_map)
                    view_masks.append(mask)
                sample_sampling_maps.append(torch.cat(view_sampling_maps, dim=2))
                sample_masks.append(torch.cat(view_masks, dim=2))
            sampling_maps.append(torch.cat(sample_sampling_maps, dim=1))
            masks.append(torch.cat(sample_masks, dim=1))

        sampling_maps = torch.cat(sampling_maps, dim=0)
        masks = torch.cat(masks, dim=0)

        return sampling_maps, masks


class DepthVolume1D(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=False, n_feats=32):
        super().__init__(depth_start, depth_end, depth_num, memory_saving)

        self.n_feats = n_feats

        self.cell0 = UNet(in_channels=n_feats + 2, enc_channels=[8, 16, 32, 64], dec_channels=[32, 16, 8], n_enc_convs=1, n_dec_convs=1)
        self.cell1 = DynamicConv2d(in_channels=7, out_channels=1, kernel_size=5, bias=True)

    def forward(self, src_feats, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, _, height, width = src_feats.shape

        # sampling_maps: [N, V, D, 2, H, W], view_masks: [N, V, D, 1, H, W]
        sampling_maps, _ = self.compute_sampling_maps(n_samples, n_views, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)

        # forward cost volume
        src_weights = []
        depth_probs = []

        for d in range(self.depth_num):
            feature_list = []
            for view in range(n_views):
                sampling_map = sampling_maps[:, view, d]
                warped_view_feature = tensor_warping(src_feats[:, view], sampling_map)
                feature_list.append(warped_view_feature)

            warped_feats = torch.stack(feature_list, dim=0)  # src_features: [V, N, C, H, W]
            
            cost = torch.einsum('vcnhw, cmnhw->vmnhw', warped_feats.permute([0,2,1,3,4]), warped_feats.permute([2,0,1,3,4]))
            view_cost = torch.mean(cost, dim=0, keepdims=True)
            view_cost_mean = torch.mean(view_cost, dim=1, keepdims=True).repeat(1, n_views, 1, 1, 1)
            warped_cost = torch.cat([warped_feats.permute(1, 0, 2, 3, 4), view_cost_mean, view_cost], dim=2)
            warped_cost = warped_cost.view(n_samples * n_views, -1, height, width)

            feature_out0 = self.cell0(warped_cost)
            
            src_weight = feature_out0[:, 0, ...].view(n_samples, n_views, height, width)
            probs = self.cell1(torch.mean(feature_out0[:, 1:, ...].view(n_samples, n_views, -1, height, width), dim=1))

            src_weights.append(src_weight)
            depth_probs.append(probs)

        src_weights = torch.stack(src_weights, dim=0).permute(1, 2, 0, 3, 4)  # [N, V, D, H, W]
        depth_probs = torch.stack(depth_probs, dim=1)  # [N, D, 1, H, W]

        return depth_probs, src_weights


class DepthVolume2DS(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=False, n_feats=32):
        super().__init__(depth_start, depth_end, depth_num, memory_saving)

        self.n_feats = n_feats

        self.deconv2 = DeconvGroupNorm(4, 3, kernel_size=16, stride=2)
        self.deconv3 = DeconvGroupNorm(4, 3, kernel_size=16, stride=2)

        self.shallow_feature_extractor = SNetDS2BN_base_8(in_channels=3)

        self.cell0 = ConvGRU2d(in_channels=18, out_channels=8)
        self.cell1 = ConvGRU2d(in_channels=8, out_channels=4)
        self.cell2 = ConvGRU2d(in_channels=4, out_channels=4)
        self.cell3 = ConvGRU2d(in_channels=7, out_channels=4)
        self.cell4 = ConvGRU2d(in_channels=8, out_channels=4)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DynamicConv2d(in_channels=11, out_channels=9, act=None, norm_cfg=None)
        self.conv4 = DynamicConv2d(in_channels=4, out_channels=1, act=None, norm_cfg=None)

    def forward(self, src_feats, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, _, height, width = src_feats.shape

        # sampling_maps: [N, V, D, 2, H, W], view_masks: [N, V, D, 1, H, W]
        sampling_maps, _ = self.compute_sampling_maps(n_samples, n_views, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)

        # forward cost volume
        src_weights = []
        depth_probs = []

        initial_state0 = torch.zeros((n_samples * n_views, 8, height, width), device=src_feats.device)
        initial_state1 = torch.zeros((n_samples * n_views, 4, height // 2, width // 2), device=src_feats.device)
        initial_state2 = torch.zeros((n_samples * n_views, 4, height // 4, width // 4), device=src_feats.device)
        initial_state3 = torch.zeros((n_samples * n_views, 4, height // 2, width // 2), device=src_feats.device)
        initial_state4 = torch.zeros((n_samples, 4, height, width), device=src_feats.device)

        for d in range(self.depth_num):
            feature_list = []
            for view in range(n_views):
                sampling_map = sampling_maps[:, view, d]
                warped_view_feature = tensor_warping(src_feats[:, view], sampling_map)
                feature_list.append(warped_view_feature)
            
            warped_feats = torch.stack(feature_list, dim=0)  # src_features: [V, N, C, H, W]

            # compute similarity, corresponds to Eq.(5) in the paper
            # cost: [V, V, N, H, W], view_cost: [1, V, N, H, W]
            cost = torch.einsum('vcnhw, cmnhw->vmnhw', warped_feats.permute([0,2,1,3,4]), warped_feats.permute([2,0,1,3,4]))
            view_cost = torch.mean(cost, dim=0, keepdims=True)
    
            # Construct input to our Souce-view Visibility Estimation (SVE) module. Corresponds to Eq.(6) in the paper
            # view_cost_mean: [1, V, N, H, W]
            view_cost_mean = torch.mean(view_cost, dim=1, keepdims=True).repeat(1, n_views, 1, 1, 1)
            view_cost = torch.cat([warped_feats.permute(1, 0, 2, 3, 4), view_cost.permute(2, 1, 0, 3, 4), view_cost_mean.permute(2, 1, 0, 3, 4)], dim=2)
            view_cost = view_cost.view(n_samples * n_views, -1, height, width)
            
            # ================ starts Source-view Visibility Estimation (SVE) ===================================
            feature_out0 = self.cell0(view_cost, initial_state0)
            initial_state0 = feature_out0
            feature_out1 = self.maxpool(feature_out0)

            feature_out1 = self.cell1(feature_out1, initial_state1)
            initial_state1 = feature_out1
            feature_out2 = self.maxpool(feature_out1)

            feature_out2 = self.cell2(feature_out2, initial_state2)
            initial_state2 = feature_out2
            feature_out2 = self.deconv2(feature_out2)
            feature_out2 = same_padding(feature_out2, feature_out1)
            feature_out2 = torch.cat([feature_out2, feature_out1], dim=1)

            feature_out3 = self.cell3(feature_out2, initial_state3)
            initial_state3 = feature_out3
            feature_out3 = self.deconv3(feature_out3)
            feature_out3 = same_padding(feature_out3, feature_out0)
            feature_out3 = torch.cat([feature_out3, feature_out0], dim=1)
            feature_out3 = self.conv3(feature_out3)
                
            # ================ ends Source-view Visibility Estimation (SVE) ===================================
            # process output:
            feature_out3 = feature_out3.view(n_samples, n_views, 9, height, width)
            src_weight = feature_out3[:, :, 0, ...]
            # The first output channel is to compute the source view visibility (ie, weight)
            feature_out3 = torch.mean(feature_out3[:, :, 1:, ...], dim=1)
            # The last eight channels are used to compute the consensus volume
            # Correspoding to Eq.(7) in the paper

            # ================ starts Soft Ray-Casting (SRC) ========================
            feature_out4 = self.cell4(feature_out3, initial_state4)
            initial_state4 = feature_out4
            features = self.conv4(feature_out4)
            # ================ ends Soft Ray-Casting (SRC) ==========================

            src_weights.append(src_weight)
            depth_probs.append(features)

        src_weights = torch.stack(src_weights, dim=0).permute(1, 2, 0, 3, 4)  # [N, V, D, H, W]
        depth_probs = torch.stack(depth_probs, dim=1)  # [N, D, 1, H, W]

        return depth_probs, src_weights


class DepthVolume2D(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=True):
        super().__init__(depth_start, depth_end, depth_num, memory_saving)

        self.deconv2 = DeconvGroupNorm(4, 3, kernel_size=16, stride=2)
        self.deconv3 = DeconvGroupNorm(4, 3, kernel_size=16, stride=2)

        self.shallow_feature_extractor = SNetDS2BN_base_8(in_channels=3)

        self.cell0 = ConvGRU2d(in_channels=18, out_channels=8)
        self.cell1 = ConvGRU2d(in_channels=8, out_channels=4)
        self.cell2 = ConvGRU2d(in_channels=4, out_channels=4)
        self.cell3 = ConvGRU2d(in_channels=7, out_channels=4)
        self.cell4 = ConvGRU2d(in_channels=8, out_channels=4)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DynamicConv2d(in_channels=11, out_channels=9, act=None, norm_cfg=None)
        self.conv4 = DynamicConv2d(in_channels=4, out_channels=1, act=None, norm_cfg=None)

    def forward(self, src_images, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, n_channels, height, width = src_images.shape
        
        old_height, old_width = 0, 0
        if self.memory_saving:
            old_height, old_width = height, width
            
            height, width = height // 2, width // 2
            src_images = nn.functional.interpolate(src_images.view(n_samples * n_views, n_channels, old_height, old_width), size=(width, height))
            src_images = src_images.view(n_samples, n_views, n_channels, height, width)
            
            dst_intrinsics[:, :, 0, 0] = dst_intrinsics[:, :, 0, 0] / 2  # N, V, 4, 4
            dst_intrinsics[:, :, 1, 1] = dst_intrinsics[:, :, 1, 1] / 2  # N, V, 4, 4
            dst_intrinsics[:, :, 0, 2] = dst_intrinsics[:, :, 0, 2] / 2  # N, V, 4, 4
            dst_intrinsics[:, :, 1, 2] = dst_intrinsics[:, :, 1, 2] / 2  # N, V, 4, 4
            src_intrinsics[:, :, 0, 0] = src_intrinsics[:, :, 0, 0] / 2  # N, V, 4, 4
            src_intrinsics[:, :, 1, 1] = src_intrinsics[:, :, 1, 1] / 2  # N, V, 4, 4
            src_intrinsics[:, :, 0, 2] = src_intrinsics[:, :, 0, 2] / 2  # N, V, 4, 4
            src_intrinsics[:, :, 1, 2] = src_intrinsics[:, :, 1, 2] / 2  # N, V, 4, 4

        # extract source view features for cost aggregation, and source view weights calculation
        view_towers = []
        for view in range(n_views):
            view_image = src_images[:, view, :, :, :]
            view_towers.append(self.shallow_feature_extractor(view_image))

        # sampling_maps: [N, V, D, 2, H, W], view_masks: [N, V, D, 1, H, W]
        sampling_maps, view_masks = self.compute_sampling_maps(n_samples, n_views, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)

        # sampling_maps:
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)

        # forward cost volume
        src_weights = []
        depth_probs = []

        initial_state0 = torch.zeros((n_samples * n_views, 8, height, width), device=src_images.device)
        initial_state1 = torch.zeros((n_samples * n_views, 4, height // 2, width // 2), device=src_images.device)
        initial_state2 = torch.zeros((n_samples * n_views, 4, height // 4, width // 4), device=src_images.device)
        initial_state3 = torch.zeros((n_samples * n_views, 4, height // 2, width // 2), device=src_images.device)
        initial_state4 = torch.zeros((n_samples, 4, height, width), device=src_images.device)

        for d in range(self.depth_num):
            feature_list = []
            for view in range(n_views):
                sampling_map = sampling_maps[:, view, d, :, :, :]
                warped_view_feature = tensor_warping(view_towers[view], sampling_map)
                feature_list.append(warped_view_feature)
    
            src_features = torch.stack(feature_list, dim=0)  # src_features: [V, N, C, H, W]

            # compute similarity, corresponds to Eq.(5) in the paper
            # cost: [V, V, N, H, W], view_cost: [1, V, N, H, W]
            cost = torch.einsum('vcnhw, cmnhw->vmnhw', src_features.permute([0,2,1,3,4]), src_features.permute([2,0,1,3,4]))
            view_cost = torch.mean(cost, dim=0, keepdims=True)
    
            # Construct input to our Souce-view Visibility Estimation (SVE) module. Corresponds to Eq.(6) in the paper
            # view_cost_mean: [1, V, N, H, W]
            view_cost_mean = torch.mean(view_cost, dim=1, keepdims=True).repeat(1, n_views, 1, 1, 1)
            view_cost = torch.cat([src_features.permute(1, 0, 2, 3, 4), view_cost.permute(2, 1, 0, 3, 4), view_cost_mean.permute(2, 1, 0, 3, 4)], dim=2)
            view_cost = view_cost.view(n_samples * n_views, -1, height, width)
            
            # ================ starts Source-view Visibility Estimation (SVE) ===================================
            feature_out0 = self.cell0(view_cost, initial_state0)
            initial_state0 = feature_out0
            feature_out1 = self.maxpool(feature_out0)

            feature_out1 = self.cell1(feature_out1, initial_state1)
            initial_state1 = feature_out1
            feature_out2 = self.maxpool(feature_out1)

            feature_out2 = self.cell2(feature_out2, initial_state2)
            initial_state2 = feature_out2
            feature_out2 = self.deconv2(feature_out2)
            feature_out2 = same_padding(feature_out2, feature_out1)
            feature_out2 = torch.cat([feature_out2, feature_out1], dim=1)

            feature_out3 = self.cell3(feature_out2, initial_state3)
            initial_state3 = feature_out3
            feature_out3 = self.deconv3(feature_out3)
            feature_out3 = same_padding(feature_out3, feature_out0)
            feature_out3 = torch.cat([feature_out3, feature_out0], dim=1)
            feature_out3 = self.conv3(feature_out3)
                
            # ================ ends Source-view Visibility Estimation (SVE) ===================================
            # process output:
            feature_out3 = feature_out3.view(n_samples, n_views, 9, height, width)
            src_weight = feature_out3[:, :, 0, ...]
            # The first output channel is to compute the source view visibility (ie, weight)
            feature_out3 = torch.mean(feature_out3[:, :, 1:, ...], dim=1)
            # The last eight channels are used to compute the consensus volume
            # Correspoding to Eq.(7) in the paper

            # ================ starts Soft Ray-Casting (SRC) ========================
            feature_out4 = self.cell4(feature_out3, initial_state4)
            initial_state4 = feature_out4
            features = self.conv4(feature_out4)
            # ================ ends Soft Ray-Casting (SRC) ==========================

            src_weights.append(src_weight)
            depth_probs.append(features)

        src_weights = torch.stack(src_weights, dim=0).permute(1, 2, 0, 3, 4)  # [N, V, D, H, W]
        depth_probs = torch.stack(depth_probs, dim=1)  # [N, D, 1, H, W]

        if self.memory_saving:
            src_weights = nn.functional.interpolate(src_weights.reshape(n_samples * n_views, -1, height, width), size=(old_width, old_height))
            src_weights = src_weights.view(n_samples, n_views, -1, old_height, old_width)

            depth_probs = nn.functional.interpolate(depth_probs.reshape(n_samples * self.depth_num, -1, height, width), size=(old_width, old_height))
            depth_probs = depth_probs.view(n_samples, self.depth_num, -1, old_height, old_width)

        return depth_probs, src_weights


def test():
    depth_volume_model = DepthVolume2D(depth_start=0.5, depth_end=10, depth_num=48)

    src_images = torch.zeros((1, 4, 3, 200, 200))
    dst_intrinsics = torch.from_numpy(
        np.array([[[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]]]))
    dst_intrinsics = dst_intrinsics.view(1, 4, 4, 4).float()
    
    src_intrinsics = torch.from_numpy(
        np.array([[[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]]]))
    src_intrinsics = src_intrinsics.view(1, 4, 4, 4).float()
    
    dst_extrinsics = np.array([[0.369466, 0.113037, -0.922344, 3.848802], [0.928934, -0.070575, 0.363457, 2.352613], [-0.024011, -0.991081, -0.131079, 1.420527], [0.000000, 0.000000, 0.000000, 1.000000]])
    dst_extrinsics = np.linalg.inv(dst_extrinsics)
    dst_extrinsics = torch.from_numpy(dst_extrinsics).view(1, 4, 4).repeat(4, 1, 1).view(1, 4, 4, 4).float()

    src_extrinsic_1 = np.array([[0.350988, 0.049005, -0.935097, 3.766715], [0.934441, -0.082568, 0.346415, 2.407103], [-0.060234, -0.995380, -0.074773, 1.435615], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_2 = np.array([[0.327968, 0.044623, -0.943634, 3.820534], [0.943492, -0.065749, 0.324810, 2.387508], [-0.047549, -0.996838, -0.063665, 1.462365], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_3 = np.array([[0.357426, 0.161129, -0.919937, 3.862065], [0.933725, -0.082870, 0.348269, 2.340595], [-0.020119, -0.983448, -0.180070, 1.393208], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_4 = np.array([[0.267619, 0.219644, -0.938156, 3.879654], [0.963498, -0.068228, 0.258874, 2.366085], [-0.007149, -0.973192, -0.229886, 1.356197], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_1 = np.linalg.inv(src_extrinsic_1)
    src_extrinsic_2 = np.linalg.inv(src_extrinsic_2)
    src_extrinsic_3 = np.linalg.inv(src_extrinsic_3)
    src_extrinsic_4 = np.linalg.inv(src_extrinsic_4)
    src_extrinsics = np.array([src_extrinsic_1, src_extrinsic_2, src_extrinsic_3, src_extrinsic_4])
    src_extrinsics = torch.from_numpy(src_extrinsics).view(1, 4, 4, 4).float()

    depth_probs, src_weights = depth_volume_model(src_images, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
    print(depth_probs.shape, src_weights.shape)
