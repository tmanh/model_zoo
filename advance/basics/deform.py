import torch
import torch.nn as nn


class DeformableModule(nn.Module):
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
    def get_interpolation_functions():
        floor, ceil = torch.floor, lambda x: torch.floor(x) + 1
        return (
            (floor, floor),
            (floor, ceil),
            (ceil, floor),
            (ceil, ceil),
        )

    @staticmethod
    def get_position(n_samples, n_offsets, n_channels, height, width, device):
        h_pos, w_pos = torch.meshgrid(torch.arange(start=0, end=height, device=device), torch.arange(start=0, end=width, device=device))
        h_pos = h_pos.view(1, 1, 1, height, width).expand(n_samples, n_offsets, n_channels, height, width).long()
        w_pos = w_pos.view(1, 1, 1, height, width).expand(n_samples, n_offsets, n_channels, height, width).long()

        sample_idx = torch.arange(n_samples, device=device).view(n_samples, 1, 1, 1, 1).expand(n_samples, n_offsets, n_channels, height, width).long()
        channels_idx = torch.arange(n_channels, device=device).view(1, 1, n_channels, 1, 1).expand(n_samples, n_offsets, n_channels, height, width).long()
        view_idx = torch.arange(n_offsets, device=device).view(1, n_offsets, 1, 1, 1).expand(n_samples, n_offsets, n_channels, height, width).long()

        return h_pos, w_pos, sample_idx, channels_idx, view_idx

    @staticmethod
    def select_by_index(images, ib, ic, ii, ih, iw):
        _, _, _, height, width = images.size()

        mask_outside = torch.bitwise_or((ih < 0), (ih >= height))
        mask_outside = torch.bitwise_or(mask_outside, (iw < 0))
        mask_outside = torch.bitwise_or(mask_outside, (iw >= width))

        mask_outside = torch.bitwise_not(mask_outside)

        ih *= mask_outside
        iw *= mask_outside

        return images[ib, ii, ic, ih, iw] * mask_outside

    def interpolate(self, images, offsets):
        n_samples, _, n_channels, height, width = images.shape
        n_offsets = offsets.shape[1] // 2
        device = images.device

        offsets = offsets.view(n_samples, 2 * n_offsets, 1, height, width)
        output = torch.zeros(n_samples, n_offsets, n_channels, height, width, device=device)

        h_pos, w_pos, sample_idx, channels_idx, view_idx = self.get_position(n_samples, n_offsets, n_channels, height, width, device)

        for fh, fw in self.get_interpolation_functions():
            ih, iw, ph, pw = self.get_trilinear_idx(offsets, h_pos, w_pos, fh, fw)
            step_output = self.select_by_index(images, sample_idx, channels_idx, view_idx, ih, iw)
            output += step_output * ph * pw

        return output
