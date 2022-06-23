import torch
import torch.nn as nn
import torch.nn.functional as functional

from .. basics.dynamic_conv import DynamicConv2d


class ProjectLayer(nn.Module):
    def __init__(self):
        super(ProjectLayer, self).__init__()

    def forward(self, input_features, project_map):
        return input_features[:, :, project_map[:, :, 0].long(), project_map[:, :, 1].long()]


class BasePos2Weight(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3, out_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        if self.out_channels is None:
            self.out_channels = 1

        self.meta_block = None

    def forward(self, x):
        return self.meta_block(x)


class Pos2Weight(BasePos2Weight):
    def __init__(self, in_channels=3, kernel_size=3, out_channels=1):
        super().__init__(in_channels, kernel_size, out_channels)

        self.meta_block = nn.Sequential(
            nn.Linear(in_channels, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, out_channels), nn.Dropout(0.25))
            

class KernelPos2Weight(BasePos2Weight):
    def __init__(self, in_channels=3, kernel_size=3, out_channels=1):
        super().__init__(in_channels, kernel_size, out_channels)

        self.meta_block = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.kernel_size * self.kernel_size * self.in_channels * self.out_channels))


class MapNet(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(MapNet, self).__init__()

        self.P2W = KernelPos2Weight(in_channels=3, out_channels=out_channels)
        self.project = ProjectLayer()

    def forward(self, pos_mat, mapping_mat, feats, depth):
        _, _, height, width = feats.shape
        n_samples, out_height, out_width, _ = pos_mat.shape

        local_weight = self.compute_local_weight(pos_mat, out_height, out_width)
        features = self.upscale_feature_maps(n_samples, height, width, mapping_mat, feats, out_height, out_width)
        return self.compute_output(n_samples, out_height, out_width, local_weight, features)

    def upscale_feature_maps(self, n_samples, height, width, mapping_mat, feats, out_height, out_width):
        cols = functional.unfold(feats, 3, padding=1)
        cols = cols.contiguous().view(n_samples, -1, height, width)

        features = cols[:, :, mapping_mat[:, :, 0], mapping_mat[:, :, 1]]
        features = features.contiguous().view(n_samples, features.size(1), 1, out_height * out_width)
        features = features.permute((0, 3, 2, 1)).contiguous()
        return features

    def compute_local_weight(self, pos_mat, out_height, out_width):
        local_weight = self.P2W(pos_mat.view(1 * out_height * out_width, -1))
        local_weight = local_weight.contiguous().view(1, out_height * out_width, -1, 1)
        return local_weight

    def compute_output(self, n_samples, out_height, out_width, local_weight, features):
        out = torch.matmul(features, local_weight)
        out = out.permute((0, 3, 1, 2))
        out = out.contiguous().view((n_samples, 1, out_height, out_width))
        return out


class ResidualMapNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.P2W = Pos2Weight(in_channels=3, out_channels=in_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pos_mat, feats, depth, depth_bicubic, intermediate=False):
        out_height, out_width = depth_bicubic.shape[-2:]
        n_samples = depth.shape[0]

        local_weight = self.compute_local_weight(pos_mat, out_height, out_width)

        up_feats = functional.interpolate(feats, size=depth_bicubic.shape[-2:], mode='bilinear', align_corners=True) + depth_bicubic.view(n_samples, -1, out_height, out_width)

        if intermediate:
            return self.compute_final_depth_map(n_samples, out_height, out_width, local_weight, up_feats), up_feats
        return self.compute_final_depth_map(n_samples, out_height, out_width, local_weight, up_feats)

    def compute_final_depth_map(self, n_samples, out_height, out_width, local_weight, up_feats):
        out = up_feats * local_weight.view(1, -1, out_height, out_width)
        return torch.sum(out, dim=1, keepdim=True).view(n_samples, -1, out_height, out_width)

    def compute_local_weight(self, pos_mat, out_height, out_width):
        local_weight = self.P2W(pos_mat.view(1 * out_height * out_width, -1))
        local_weight = local_weight.view(1, out_height, out_width, self.P2W.out_channels)
        local_weight = local_weight.permute(0, 3, 1, 2)
        return self.softmax(local_weight)
