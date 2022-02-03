import torch
import torch.nn.functional

from torch import nn


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
    def __init__(self, in_channels=1, kernel_size=3, out_channels=1):
        super().__init__(in_channels, kernel_size, out_channels)

        self.meta_block = nn.Sequential(
            nn.Linear(3, 256), nn.ReLU(),
            nn.Linear(256, self.in_channels), nn.ReLU())


class KernelPos2Weight(BasePos2Weight):
    def __init__(self, in_channels, kernel_size=3, out_channels=1):
        super().__init__(in_channels, kernel_size, out_channels)

        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.kernel_size * self.kernel_size * self.in_channels * self.out_channels))


class MapNet(nn.Module):
    def __init__(self, in_channels):
        super(MapNet, self).__init__()

        self.P2W = KernelPos2Weight(in_channels=in_channels)
        self.project = ProjectLayer()

    def forward(self, pos_mat, mapping_mat, feats, depth):
        _, _, height, width = feats.shape
        n_samples, out_height, out_width, _ = pos_mat.shape

        local_weight = self.compute_local_weight(n_samples, pos_mat, out_height, out_width)
        features = self.upscale_feature_maps(n_samples, height, width, mapping_mat, feats, out_height, out_width)
        return self.compute_output(n_samples, out_height, out_width, local_weight, features)

    def upscale_feature_maps(self, n_samples, height, width, mapping_mat, feats, out_height, out_width):
        cols = nn.functional.unfold(feats, 3, padding=1)
        cols = cols.contiguous().view(n_samples, -1, height, width)

        features = cols[:, :, mapping_mat[:, :, 0], mapping_mat[:, :, 1]]
        features = features.contiguous().view(n_samples, features.size(1), 1, out_height * out_width)
        features = features.permute((0, 3, 2, 1)).contiguous()
        return features

    def compute_local_weight(self, n_samples, pos_mat, out_height, out_width):
        local_weight = self.P2W(pos_mat.view(n_samples * out_height * out_width, -1))
        local_weight = local_weight.contiguous().view(n_samples, out_height * out_width, -1, 1)
        return local_weight

    def compute_output(self, n_samples, out_height, out_width, local_weight, features):
        out = torch.matmul(features, local_weight)
        out = out.permute((0, 3, 1, 2))
        out = out.contiguous().view((n_samples, 1, out_height, out_width))
        return out


class ResidualMapNet(nn.Module):
    def __init__(self, in_channels):
        super(ResidualMapNet, self).__init__()

        self.P2W = Pos2Weight(in_channels=in_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pos_mat, mapping_mat, feats, depth, intermediate=False):
        n_samples, n_feats, _, _ = feats.size()
        _, out_height, out_width, _ = pos_mat.size()

        n_samples, out_height, out_width, _ = pos_mat.size()

        local_weight = self.compute_local_weight(pos_mat, n_samples, n_feats, out_height, out_width)
        up_depth = self.upscale_depth_maps(mapping_mat, feats, depth)

        if intermediate:
            return self.compute_final_depth_map(n_samples, out_height, out_width, local_weight, up_depth), up_depth
        return self.compute_final_depth_map(n_samples, out_height, out_width, local_weight, up_depth)

    def compute_final_depth_map(self, n_samples, out_height, out_width, local_weight, up_depth):
        out = up_depth * local_weight
        return torch.sum(out, dim=1).view(n_samples, 1, out_height, out_width)

    def upscale_depth_maps(self, mapping_mat, feats, depth):
        feats = feats + depth
        return feats[:, :, mapping_mat[:, :, 0], mapping_mat[:, :, 1]]

    def compute_local_weight(self, pos_mat, n_samples, n_feats, out_height, out_width):
        local_weight = self.P2W(pos_mat.view(n_samples * out_height * out_width, -1))
        local_weight = local_weight.view(n_samples, out_height, out_width, n_feats)
        local_weight = local_weight.permute(0, 3, 1, 2)
        return self.softmax(local_weight)
