import torch
import torch.nn as nn
import torch.nn.functional as functional


class MappingRNN(nn.Module):
    def __init__(self, enc_net, merge_net, merge_channels=64, freeze_enc=True, freeze_all=False, train_enhance=False):
        super().__init__()

        self.enc_net = enc_net
        self.merge_net = merge_net

        self.rgb_conv = nn.Conv2d(merge_channels, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha_conv = nn.Conv2d(merge_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)

        if freeze_enc or freeze_all:
            for param in self.enc_net.parameters():
                param.requires_grad = False
        
        if freeze_all:
            for param in self.merge_net.parameters():
                param.requires_grad = False

        self.train_enhance = train_enhance

    def compute_network_output(self, projected_depths, out_colors, alphas, preprocessing):
        valid_masks = (projected_depths > 0).float()
        valid_mask = (torch.sum(valid_masks, dim=1) > 0).float()

        output = self.compute_out_color(out_colors, alphas, preprocessing)
        output['valid_mask'] = valid_mask
        return output

    def compute_out_color(self, colors, alphas, preprocessing):
        colors = torch.stack(colors)

        if preprocessing:
            return colors.permute(1, 0, 2, 3, 4)

        alphas = torch.stack(alphas)
        alphas = torch.softmax(alphas, dim=0)

        refine = (alphas * colors).sum(dim=0)

        return {'refine': refine, 'deep_dst_color': None, 'deep_prj_colors': colors.permute(1, 0, 2, 3, 4), 'prj_colors': None, 'dst_color': None, 'valid_mask': None}

    def estimate_view_color(self, x, out_colors, alphas):
        out_colors.append(self.rgb_conv(x))
        alphas.append(self.alpha_conv(x))

    def compute_encoded_features(self, colors, depths, masks, distance_maps, sampling_maps):
        batch_size, n_views, in_channel, height, width = colors.shape

        sampling_maps = sampling_maps.reshape(batch_size * n_views, 2, height, width).permute(0, 2, 3, 1)
        colors = colors.reshape(batch_size * n_views, in_channel, height, width)

        encoded_features = self.enc_net(colors)

        projected_features = functional.grid_sample(encoded_features, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        projected_features = projected_features.view(batch_size, n_views, -1, height, width)

        projected_colors = functional.grid_sample(colors, sampling_maps.reshape(batch_size * n_views, 2, height, width).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        projected_colors = projected_colors.view(batch_size, n_views, in_channel, height, width)

        projected_features = torch.cat([projected_features, projected_colors, distance_maps, depths, masks], dim=2)

        return projected_features, projected_colors, n_views

    def compute_enhanced_images(self, projected_features, depths, n_views, preprocessing):
        hs = None
        out_colors = []
        alphas = []
        for vidx in range(n_views):
            y, hs = self.merge_net(projected_features[:, vidx], hs)
            self.estimate_view_color(y, out_colors, alphas)

        return self.compute_network_output(depths, out_colors, alphas, preprocessing)

    def forward(self, depths, colors, masks, sampling_maps, distance_maps, preprocessing):
        projected_features, _, n_views = self.compute_encoded_features(colors, depths, masks, distance_maps, sampling_maps)
        return self.compute_enhanced_images(projected_features, depths, n_views, preprocessing)
