import numpy as np

import torch
import torch.nn as nn

from .snet import SNetDS2BN_base_8
from .basics.conv_gru import ConvGRU2d
from .basics.geometry import create_sampling_map_target2source, tensor_warping
from .basics.dynamic_conv import DeconvGroupNorm, DynamicConv2d


class BaseDepthVolumeModel(nn.Module):
    def __init__(self, depth_start, depth_end, depth_num):
        super().__init__()

        self.depth_start = depth_start
        self.depth_end = depth_end
        self.depth_num = depth_num
        self.depth_ranges = np.linspace(self.depth_start, self.depth_end, self.depth_num)

    def compute_sampling_maps(self, n_samples, n_views, ys_dst, xs_dst, ys_src, xs_src, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        sampling_maps = []
        masks = []
        for n in range(n_samples):
            sample_sampling_maps = []
            sample_masks = []
            for i in range(n_views):
                view_sampling_maps = []
                view_masks = []

                for d in self.depth_ranges:
                    y_dst, x_dst, y_src, x_src = ys_dst[n, i], xs_dst[n, i], ys_src[n, i], xs_src[n, i]
                    dst_intrinsic, dst_extrinsic = dst_intrinsics[n, i, ...], dst_extrinsics[n, i, ...]
                    src_intrinsic, src_extrinsic = src_intrinsics[n, i, ...], src_extrinsics[n, i, ...]
                    sampling_map, mask = create_sampling_map_target2source(d, y_dst, x_dst, y_src, x_src, height, width, dst_intrinsic, dst_extrinsic, src_intrinsic, src_extrinsic)
                    view_sampling_maps.append(sampling_map)
                    view_masks.append(mask)
                sample_sampling_maps.append(torch.cat(view_sampling_maps, dim=2))
                sample_masks.append(torch.cat(view_masks, dim=2))
            sampling_maps.append(torch.cat(sample_sampling_maps, dim=1))
            masks.append(torch.cat(sample_masks, dim=1))

        sampling_maps = torch.cat(sampling_maps, dim=0)
        masks = torch.cat(masks, dim=0)

        return sampling_maps, masks

class DepthVolumeModel(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num):
        super().__init__(depth_start, depth_end, depth_num)

        self.depth_start = depth_start
        self.depth_end = depth_end
        self.depth_num = depth_num
        self.depth_ranges = np.linspace(self.depth_start, self.depth_end, self.depth_num)

        self.deconv2 = DeconvGroupNorm(4, 3, kernel_size=16, stride=2)
        self.deconv3 = DeconvGroupNorm(4, 3, kernel_size=16, stride=2)

        self.shallow_feature_extractor = SNetDS2BN_base_8(in_channels=3)

        self.cell0 = ConvGRU2d(in_channels=18, out_channels=8)
        self.cell1 = ConvGRU2d(in_channels=8, out_channels=4)
        self.cell2 = ConvGRU2d(in_channels=4, out_channels=4)
        self.cell3 = ConvGRU2d(in_channels=7, out_channels=4)
        self.cell4 = ConvGRU2d(in_channels=8, out_channels=4)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DynamicConv2d(in_channels=11, out_channels=9, act=None, batch_norm=False)
        self.conv4 = DynamicConv2d(in_channels=4, out_channels=1, act=None, batch_norm=False)

    def forward(self, src_images, ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, n_channels, height, width = src_images.shape

        # extract source view features for cost aggregation, and source view weights calculation
        view_towers = []
        for view in range(n_views):
            view_image = src_images[:, view, :, :, :]
            view_towers.append(self.shallow_feature_extractor(view_image))

        # sampling_maps: [N, V, D, 2, H, W], view_masks: [N, V, D, 1, H, W]
        sampling_maps, view_masks = self.compute_sampling_maps(n_samples, n_views, ys_dst, xs_dst, ys_src, xs_src, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)

        # sampling_maps:
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)

        # forward cost volume
        src_weights = []
        depth_probs = []
        warped_feature_whole = []

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
            warped_feature_whole.append(src_features)

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
            feature_out2 = torch.cat([feature_out2, feature_out1], dim=1)

            feature_out3 = self.cell3(feature_out2, initial_state3)
            initial_state3 = feature_out3
            feature_out3 = self.deconv3(feature_out3)
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


def test():
    depth_volume_model = DepthVolumeModel(depth_start=0.5, depth_end=10, depth_num=48)

    src_images = torch.zeros((1, 4, 3, 200, 200))
    ys_dst = torch.from_numpy(np.array([10, 10, 10, 10])).view(1, 4)
    xs_dst = torch.from_numpy(np.array([10, 10, 10, 10])).view(1, 4)
    ys_src = torch.from_numpy(np.array([10, 15, 5, 8])).view(1, 4)
    xs_src = torch.from_numpy(np.array([11, 13, 8, 15])).view(1, 4)
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

    depth_probs, src_weights = depth_volume_model(src_images, ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
    print(depth_probs.shape, src_weights.shape)