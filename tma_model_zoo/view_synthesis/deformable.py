import itertools
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..basics.geometry import create_sampling_map_src2tgt
from ..enhancement.dmsr import LightDMSR
from ..basics.deform import DeformableModule
from ..universal import *


class DeformableNetwork(DeformableModule):
    def __init__(self, n_images=4, n_feats=64, n_channels=4, training=True, freeze_enc=True):
        super().__init__()

        self.training = training
        self.n_feats = n_feats
        self.n_views = n_images
        self.n_channels = n_channels
        self.input_length = self.n_views * self.n_channels

        self.interpolations = self.get_interpolation_functions()
        self.compute_trilinear_idx = self.get_trilinear_idx

        self.act = nn.LeakyReLU(inplace=True)

        self.rgb_conv = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.global_conv = nn.Sequential(*[MemorySavingUNet(in_channels=70 * 4, enc_channels=[32, 64, 128, 256], dec_channels=[256, 128, 64, 32]),
                                           nn.Conv2d(32, 16, 3, 1, 1), nn.LeakyReLU(inplace=True)])

        self.offset_predictor = nn.Sequential(*[MemorySavingUNet(in_channels=64 + 70 + 16, enc_channels=[32, 64, 128, 256], dec_channels=[256, 128, 64, 32]),
                                                nn.Conv2d(32, 3, 3, 1, 1), nn.ReLU(inplace=True)])

        self.enc_net = VGGUNet()
        self.merge_net = GRUUNet(self.n_feats + 6, enc_channels=[64, 128, 256, 512], dec_channels=[256, 128, 64], n_enc_convs=3, n_dec_convs=3, act=self.act)

        self.freeze_enc = freeze_enc
        self.freeze(freeze_enc)

    def freeze(self, freeze_enc):
        if not freeze_enc:
            return
        
        for param in self.enc_net.parameters():
            param.requires_grad = False
        for param in self.rgb_conv.parameters():
            param.requires_grad = False
        for param in self.alpha_conv.parameters():
            param.requires_grad = False
        for param in self.merge_net.parameters():
            param.requires_grad = False

        for module in self.enc_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.momentum = 0
                module.eval()

        for module in self.merge_net.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.momentum = 0
                module.eval()

    def compute_encoded_features(self, colors, depths, masks, distance_maps, sampling_maps):
        batch_size, n_views, in_channel, height, width = colors.shape

        sampling_maps = sampling_maps.reshape(batch_size * n_views, 2, height, width).permute(0, 2, 3, 1)
        colors = colors.reshape(batch_size * n_views, in_channel, height, width)

        encoded_features = self.enc_net(colors)

        if self.training:
            encoded_features = [self.enc_net(colors[i:i+1]) for i in range(n_views)]
            encoded_features = torch.cat(encoded_features, dim=0)

        projected_features = functional.grid_sample(encoded_features, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        projected_features = projected_features.view(batch_size, n_views, -1, height, width)

        projected_colors = functional.grid_sample(colors, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        projected_colors = projected_colors.view(batch_size, n_views, in_channel, height, width)

        projected_features = torch.cat([projected_features, projected_colors, distance_maps, depths, masks], dim=2)

        return projected_features, projected_colors, n_views

    def compute_enhanced_images(self, projected_features, n_views):
        hs = None
        out_colors = []
        alphas = []
        feats = []

        for vidx in range(n_views):
            y, hs = self.merge_net(projected_features[:, vidx], hs)

            out_colors.append(self.rgb_conv(y))
            alphas.append(self.alpha_conv(y))
            feats.append(y)

        return self.compute_out_color(out_colors, alphas, feats)

    def compute_out_color(self, colors, alphas, feats):
        feats = torch.stack(feats).permute(1, 0, 2, 3, 4)
        colors = torch.stack(colors)
        alphas = torch.softmax(torch.stack(alphas), dim=0)
        return (alphas * colors).sum(dim=0), colors.permute(1, 0, 2, 3, 4), feats

    def forward(self, colors, projected_depths, sampling_maps, distance_maps, xs_src, ys_src, xs_dst, ys_dst,
                dst_extrinsics, src_extrinsics, dst_intrinsics, src_intrinsics):
        n_samples, n_views, _, height, width = colors.shape

        # compute masks
        valid_masks = (projected_depths > 0).float()
        valid_mask = (torch.sum(valid_masks, dim=1) > 0).float() if self.training else None

        # compute projected features and enhanced images
        projected_feats, projected_colors, n_views = self.compute_encoded_features(colors, projected_depths, valid_masks, distance_maps, sampling_maps)
        deep_image, enhanced_colors, deep_feats = self.compute_enhanced_images(projected_feats, n_views)

        if self.freeze_enc:
            global_feats = self.global_conv(projected_feats.view(n_samples, -1, height, width))
            global_feats = global_feats.view(n_samples, 1, -1, height, width).repeat(1, n_views, 1, 1, 1) 

            # compute offsets and weights
            coarse_feats = torch.cat([global_feats, projected_feats, deep_feats], dim=2).view(n_samples * n_views, -1, height, width)

            if not self.training:
                del projected_feats
                del global_feats
                del deep_feats

            offsets = self.offset_predictor(coarse_feats)

            return self.blending(deep_image, valid_mask, offsets, projected_colors, enhanced_colors, n_samples, n_views, height, width)

        return {'refine': deep_image, 'deep_dst_color': None, 'deep_prj_colors': enhanced_colors.view(n_samples, n_views, -1, height, width),
                'prj_colors': projected_colors.view(n_samples, n_views, -1, height, width), 'dst_color': None, 'valid_mask': valid_mask}

    def blending(self, deep_image, valid_mask, offsets, projected_colors, enhanced_colors, n_samples, n_views, height, width):
        weights = torch.softmax(offsets[:, 2:, :, :], dim=0).view(n_samples, n_views, -1, height, width)

        if not self.training:
            weights = self.enhance_weights(weights)

        # create the interpolation
        sampling_maps = offsets[:, :2, :, :].reshape(n_samples, -1, height, width)
        samples = self.interpolate(enhanced_colors, sampling_maps)

        # combine images
        deformable_output = self.combine_warped_images(samples, weights)

        output = {'refine': deformable_output, 'deep_dst_color': None, 'deep_prj_colors': enhanced_colors.view(n_samples, n_views, -1, height, width),
                  'prj_colors': projected_colors.view(n_samples, n_views, -1, height, width), 'dst_color': None, 'valid_mask': valid_mask}

        if self.training:
            output['prj_colors'] = None

        return output

    @staticmethod
    def combine_warped_images(samples, weights):
        return torch.sum(weights * samples, dim=1, keepdim=False)

    @staticmethod
    def enhance_weights(weights):
        squared_weights = weights ** 12
        sum_squared_weights = torch.sum(squared_weights, dim=1, keepdim=True)
        return squared_weights / sum_squared_weights


class LightDeformableNetwork(DeformableModule):
    def __init__(self, n_images=4, n_feats=64, n_channels=4, training=True, freeze_enc=False, separate=False):
        super().__init__()

        self.training = training
        self.n_feats = n_feats
        self.n_views = n_images
        self.n_channels = n_channels
        self.input_length = self.n_views * self.n_channels

        self.freeze_enc = freeze_enc
        self.separate = separate

        self.interpolations = self.get_interpolation_functions()
        self.compute_trilinear_idx = self.get_trilinear_idx

        self.act = nn.LeakyReLU(inplace=True)

        self.light_dsr = LightDMSR()
        self.merge_conv = Resnet(4, 32, 3, 3, 4, tail=True)

        self.shallow_resnet = Resnet(3, 64, 3, 3, 64)

        self.gru = ResidualGRU(69, 64, n_resblock=3)

        self.rgb_conv = nn.Conv2d(67, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha_conv = nn.Conv2d(67, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.offset_predictor = nn.Sequential(*[MemorySavingUNet(in_channels=64 + 70 + 16, enc_channels=[32, 64, 128, 256], dec_channels=[256, 128, 64, 32]),
                                                nn.Conv2d(32, 3, 3, 1, 1), nn.ReLU(inplace=True)])

    def compute_encoded_features(self, colors, depths, sampling_maps):
        batch_size, n_views, in_channel, height, width = colors.shape

        sampling_maps = sampling_maps.reshape(batch_size * n_views, 2, height, width).permute(0, 2, 3, 1)
        colors = colors.reshape(batch_size * n_views, in_channel, height, width)
        depths = depths.reshape(batch_size * n_views, 1, height, width)

        # """
        if self.training:
            encoded_features = [self.shallow_resnet(colors[i:i+1]) for i in range(n_views)]
            encoded_features = torch.cat(encoded_features, dim=0)
        else:
            encoded_features = self.shallow_resnet(colors)
        # """

        projected_depths = functional.grid_sample(depths, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        projected_depths = projected_depths.view(batch_size, n_views, -1, height, width)

        projected_colors = functional.grid_sample(colors, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        projected_colors = projected_colors.view(batch_size, n_views, in_channel, height, width)

        # compute masks
        valid_masks = (projected_depths > 0).float()
        valid_mask = (torch.sum(valid_masks, dim=1) > 0).float() if self.training else None

        projected_features = functional.grid_sample(encoded_features, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        projected_features = projected_features.view(batch_size, n_views, -1, height, width)
        projected_features = torch.cat([projected_features, projected_colors, projected_depths, valid_masks], dim=2)

        return projected_features, projected_colors, projected_depths, valid_masks, valid_mask, n_views

    def compute_global_feats(self, projected_features, n_views):
        hs = None
        feats = []
        for vidx in range(n_views):
            if self.separate:
                x, hs = self.gru(projected_features[:, vidx], hs)
                feats.append(x)
            else:
                feats, hs = self.gru(projected_features[:, vidx], hs)
        return feats

    def create_sampling_maps(self, depths, dst_extrinsics, src_extrinsics, dst_intrinsics, src_intrinsics, n_samples, n_views, height, width):
        sampling_maps = torch.zeros((n_samples, n_views, 2, height, width), device=depths.device)
        for i, j in itertools.product(range(n_samples), range(n_views)):
            sampling_map = create_sampling_map_src2tgt(depths[i, j], height, width, dst_intrinsics[i, 0], dst_extrinsics[i, 0], src_intrinsics[i, j], src_extrinsics[i, j])
            sampling_maps[i, j] = sampling_map
        return sampling_maps

    def forward(self, colors, depths, sampling_maps, distance_maps, xs_src, ys_src, xs_dst, ys_dst,
                dst_extrinsics, src_extrinsics, dst_intrinsics, src_intrinsics):
        n_samples, n_views, _, height, width = colors.shape

        prj_colors = None
        """
        prj_colors = functional.grid_sample(colors.reshape(n_samples * n_views, 3, height, width),
                                            sampling_maps.reshape(n_samples * n_views, 2, height, width).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        prj_colors = prj_colors.view(n_samples, n_views, 3, height, width)
        # """

        # depth completion
        depth_ilr, depth_feats = self.light_dsr.depth_completion(colors.view(n_samples * n_views, -1, height, width),
                                                                 depths.view(n_samples * n_views, -1, height, width),
                                                                 (depths.view(n_samples * n_views, -1, height, width) > 0).float())
        depth_ilr = depth_ilr.view(n_samples, n_views, -1, height, width)
        depth_feats = depth_feats.view(n_samples, n_views, -1, height, width)

        # create sampling maps from completed depths
        sampling_maps = self.create_sampling_maps(depth_ilr, dst_extrinsics, src_extrinsics, dst_intrinsics, src_intrinsics, n_samples, n_views, height, width)

        # compute projected features and enhanced images
        projected_feats, projected_colors, projected_depths, valid_masks, valid_mask, n_views = self.compute_encoded_features(colors, depth_ilr, sampling_maps)

        # generate weights from the projection of the completed depths
        weights = self.merge_conv(projected_depths.view(n_samples, -1, height, width)).view(n_samples, n_views, 1, height, width)
        weights = torch.softmax(weights, dim=1) * valid_masks

        merged_image = torch.sum(weights * projected_colors, dim=1) / torch.sum(weights + 1e-7, dim=1)

        return {'refine': None, 'deep_dst_color': merged_image, 'deep_prj_colors': projected_colors, 'prj_colors': prj_colors, 'dst_color': None,
                'deep_prj_depths': projected_depths, 'deep_src_depths': depth_ilr, 'src_colors': colors, 'src_depths': depths,
                'dst_intrinsic': dst_intrinsics, 'dst_extrinsic': dst_extrinsics, 'src_intrinsics': src_intrinsics, 'src_extrinsics': src_extrinsics,
                'valid_mask': valid_mask}
        global_feats = self.compute_global_feats(projected_feats, n_views)
        # exit()

        if self.separate:
            rgbs = []
            alphas = []
            for i in range(n_views):
                coarse_feats = torch.cat([global_feats[i], projected_colors[:, i]], dim=1)
                rgbs.append(self.rgb_conv(coarse_feats))
                alphas.append(self.alpha_conv(coarse_feats))
            alphas = torch.softmax(torch.stack(alphas, dim=0), dim=0)
            rgbs = torch.stack(rgbs, dim=0) + projected_colors.permute(1, 0, 2, 3, 4)
            deep_image = torch.sum(rgbs * alphas, dim=0)
            return (deep_image, rgbs.view(n_samples, n_views, -1, height, width), projected_colors.view(n_samples, n_views, -1, height, width)), valid_mask
        else:
            if self.freeze_enc:
                # compute offsets and weights
                coarse_feats = torch.cat([global_feats, projected_feats], dim=2).view(n_samples * n_views, -1, height, width)

                if not self.training:
                    del projected_feats
                    del global_feats

                offsets = self.offset_predictor(coarse_feats)

                return self.blending(valid_mask, offsets, projected_colors, n_samples, n_views, height, width)
            else:
                rgbs = []
                alphas = []
                for i in range(n_views):
                    coarse_feats = torch.cat([global_feats, projected_colors[:, i]], dim=1)
                    rgbs.append(self.rgb_conv(coarse_feats))
                    alphas.append(self.alpha_conv(coarse_feats))
                alphas = torch.softmax(torch.stack(alphas, dim=0), dim=0)
                rgbs = torch.stack(rgbs, dim=0) + projected_colors.permute(1, 0, 2, 3, 4)
                deep_image = torch.sum(rgbs * alphas, dim=0)
                return (deep_image, rgbs.view(n_samples, n_views, -1, height, width), projected_colors.view(n_samples, n_views, -1, height, width)), valid_mask

    def blending(self, valid_mask, offsets, projected_colors, n_samples, n_views, height, width):
        weights = torch.softmax(offsets[:, 2:, :, :], dim=0).view(n_samples, n_views, -1, height, width)
        
        if not self.training:
            weights = self.enhance_weights(weights)

        # create the interpolation
        sampling_maps = offsets[:, :2, :, :].reshape(n_samples, -1, height, width)
        weights = offsets[:, 2:, :, :].reshape(n_samples, -1, height, width)

        # combine images
        samples = functional.grid_sample(projected_colors, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        deformable_output = self.combine_warped_images(samples, weights)

        return (deformable_output,), valid_mask
        # return (deformable_output,projected_colors.view(n_samples, n_views, -1, height, width)), valid_mask

    @staticmethod
    def combine_warped_images(samples, weights):
        return torch.sum(weights * samples, dim=1, keepdim=False)
