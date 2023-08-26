import torch
import torch.nn as nn
import torch.nn.functional as functional

from tma_model_zoo.universal.swin import SwinTransformerBlock


class ConvGRU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, act=torch.tanh, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_gates = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=2 * out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_can = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.act = act

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)

        combined = torch.cat([x, h], dim=1)
        combined_conv = torch.sigmoid(self.conv_gates(combined))

        r = combined_conv[:, :self.out_channels]
        z = combined_conv[:, self.out_channels:]

        combined = torch.cat([x, r * h], dim=1)
        n = self.act(self.conv_can(combined))

        return z * h + (1 - z) * n


class SwinConvGRU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, act=torch.tanh, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        window_size = 4
        num_heads = out_channels // 32

        self.swin = SwinTransformerBlock(dim=out_channels,
                             num_heads=num_heads, window_size=window_size,
                             shift_size=0,
                             mlp_ratio=4.0,
                             qkv_bias=True,
                             drop=0.05, attn_drop=0.05,
                             drop_path=0.05,
                             norm_layer=nn.LayerNorm,
                             pretrained_window_size=False)

        self.conv_gate = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=2 * out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_dgate = nn.Conv2d(in_channels=(in_channels + out_channels) // 4, out_channels=2 * out_channels, kernel_size=1, padding=0, bias=bias)
        self.conv_can = nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.act = act

    def forward(self, c, d, ch, dh):
        if ch is None:
            ch = torch.zeros((c.shape[0], self.out_channels, c.shape[2], c.shape[3]), dtype=c.dtype, device=c.device)
            dh = torch.zeros((d.shape[0], self.out_channels // 4, d.shape[2], d.shape[3]), dtype=c.dtype, device=c.device)

        combined = torch.cat([c, ch], dim=1)
        depth_combined = torch.cat([d, dh], dim=1)
        gating = torch.sigmoid(self.conv_gate(combined * self.conv_dgate(depth_combined)))

        r = gating[:, :self.out_channels]
        z = gating[:, self.out_channels:]

        combined_feats = torch.cat([c, r * ch], dim=1)

        N, _, H, W = combined.shape
        combined_feats = self.act(self.conv_can(combined_feats))

        n = self.swin(combined_feats.permute(0, 2, 3, 1).view(N, H * W, -1), combined.shape[-2:])
        n = n.view(N, H, W, -1).permute(0, 3, 1, 2)

        return z * ch + (1 - z) * n


class LightGRU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, act=torch.tanh, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_gates = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=2, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_can = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.act = act

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)

        combined = torch.cat([x, h], dim=1)
        combined_conv = torch.sigmoid(self.conv_gates(combined))
        del combined

        r = combined_conv[:, :1]
        z = combined_conv[:, 1:]

        combined = torch.cat([x, r * h], dim=1)
        n = self.act(self.conv_can(combined))
        del combined

        return z * h + (1 - z) * n


class DeformableLightGRU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, act=torch.tanh, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_gates = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=4, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_can = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.act = act
        self.interpolations = self.get_interpolation_functions()

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)

        _, _, height, width = x.shape
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv_gates(combined)

        r = torch.sigmoid(combined_conv[:, :1])
        z = torch.sigmoid(combined_conv[:, 1:2])

        sampling_maps = combined_conv[:, 2:].permute(0, 2, 3, 1)

        # shifted_h = self.interpolate(h, combined_conv[:, 2:])
        h_pos, w_pos = torch.meshgrid(torch.arange(start=0, end=height, device=x.device), torch.arange(start=0, end=width, device=x.device))
        h_pos = h_pos / (height / 2) - 1
        h_pos = h_pos / (width / 2) - 1
        base_sampling = torch.cat([w_pos.view(1, height, width, 1), h_pos.view(1, height, width, 1)], dim=3)
        sampling_maps += base_sampling
        shifted_h = nn.functional.grid_sample(h, sampling_maps, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        combined = torch.cat([x, r * shifted_h], dim=1)
        n = self.act(self.conv_can(combined))

        return z * shifted_h + (1 - z) * n

    def interpolate(self, features, offsets):
        n_samples, n_channels, height, width = features.shape

        output = torch.zeros((n_samples, n_channels, height, width), device=features.device)

        h_pos, w_pos, sample_idx, channels_idx = self.get_position(n_samples, n_channels, height, width, features.device)

        for fh, fw in self.interpolations:
            ih, iw, ph, pw = self.get_trilinear_idx(offsets, h_pos, w_pos, fh, fw)
            step_output = self.select_by_index(features, sample_idx, channels_idx, ih, iw)
            output += step_output * ph * pw
            
        return output

    def select_by_index(self, features, ib, ic, ih, iw):
        _, _, height, width = features.shape

        mask_outside = torch.bitwise_not((ih < 0) | (ih >= height) | (iw < 0) | (iw >= width))

        ih *= mask_outside
        iw *= mask_outside

        return features[ib, ic, ih, iw] * mask_outside

    @staticmethod
    def get_interpolation_functions():
        floor, ceil = torch.floor, lambda x: torch.floor(x) + 1
        return (
            (floor, floor),
            (floor, ceil),
            (ceil, floor),
            (ceil, ceil),
        )

    @staticmethod
    def get_trilinear_idx(offsets, h_pos, w_pos, fh, fw):
        fh_position = fh(offsets[:, 0::2, ...])
        fw_position = fw(offsets[:, 1::2, ...])

        ih = fh_position.long() + h_pos
        iw = fw_position.long() + w_pos

        ph = (1 - torch.abs(fh_position - offsets[:, 0::2, ...]))
        pw = (1 - torch.abs(fw_position - offsets[:, 1::2, ...]))

        return ih, iw, ph, pw

    @staticmethod
    def get_position(n_samples, n_channels, height, width, device):
        h_pos, w_pos = torch.meshgrid(torch.arange(start=0, end=height, device=device), torch.arange(start=0, end=width, device=device))
        h_pos = h_pos.view(1, 1, height, width).expand(n_samples, n_channels, height, width).long()
        w_pos = w_pos.view(1, 1, height, width).expand(n_samples, n_channels, height, width).long()

        sample_idx = torch.arange(n_samples, device=device).view(n_samples, 1, 1, 1).expand(n_samples, n_channels, height, width).long()
        channels_idx = torch.arange(n_channels, device=device).view(1, n_channels, 1, 1).expand(n_samples, n_channels, height, width).long()
        
        return h_pos, w_pos, sample_idx, channels_idx
