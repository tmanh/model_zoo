import torch
import torch.nn as nn
import torch.nn.functional as functional


def create_loc_matrix(depth_value, height, width, device):
    y = torch.arange(start=0, end=int(height), device=device).view(1, height, 1).repeat(1, 1, width)       # columns
    x = torch.arange(start=0, end=int(width), device=device).view(1, 1, width).repeat(1, height, 1)        # rows
    ones = torch.ones((1, height, width), device=device)
    z = depth_value * ones

    return torch.cat([x * z, y * z, z, ones], dim=0).view((4, -1))


def create_sampling_map_target2source(depth_value, height, width, dst_intrinsic, dst_extrinsic, src_intrinsic, src_extrinsic):
    # compute location matrix
    pos_matrix = create_loc_matrix(depth_value, height, width, dst_intrinsic.device).reshape(4, -1)
    pos_matrix = torch.linalg.inv((dst_intrinsic @ dst_extrinsic)) @ pos_matrix
    pos_matrix = src_intrinsic @ src_extrinsic @ pos_matrix
    pos_matrix = pos_matrix.reshape((4, height, width))

    # compute sampling maps
    sampling_map = pos_matrix[:2, :, :] / (pos_matrix[2:3, :, :] + 1e-7)

    # compute mask
    mask0 = (sampling_map[0:1, ...] >= 0).float()
    mask1 = (sampling_map[0:1, ...] < width).float()
    mask2 = (sampling_map[1:2, ...] >= 0).float()
    mask3 = (sampling_map[1:2, ...] <= height).float()
    mask = mask0 * mask1 * mask2 * mask3  # indicator of valid value (1: valid, 0: invalid)

    # normalize
    sampling_map[0, :, :] = (sampling_map[0, :, :] / width) * 2 - 1
    sampling_map[1, :, :] = (sampling_map[1, :, :] / height) * 2 - 1

    return sampling_map.reshape((1, 1, 1, 2, height, width)), mask.reshape((1, 1, 1, 1, height, width))


def create_sampling_map_src2tgt(depth_value, height, width, dst_intrinsic, dst_extrinsic, src_intrinsic, src_extrinsic):
    # compute location matrix
    pos_matrix = create_loc_matrix(depth_value, 0, 0, height, width, dst_intrinsic.device).reshape(4, -1)
    pos_matrix = torch.linalg.inv((src_intrinsic @ src_extrinsic)) @ pos_matrix
    pos_matrix = dst_intrinsic @ dst_extrinsic @ pos_matrix
    pos_matrix = pos_matrix.reshape((4, height, width))

    # compute sampling maps
    sampling_map = 15000 * torch.ones((2, height, width), device=dst_intrinsic.device)
    locations = (pos_matrix[:2, :, :] / (pos_matrix[2:3, :, :] + 1e-7)).long()

    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.cat([grid_x.view(1, height, width), grid_y.view(1, height, width)], dim=0).float().to(dst_intrinsic.device)

    # compute mask
    mask0 = (locations[0:1, ...] >= 0).float()
    mask1 = (locations[0:1, ...] < width).float()
    mask2 = (locations[1:2, ...] >= 0).float()
    mask3 = (locations[1:2, ...] < height).float()
    mask = (mask0 * mask1 * mask2 * mask3).bool().view(1, height, width).repeat(2, 1, 1)  # indicator of valid value (1: valid, 0: invalid)

    selected_locations = locations[mask].view(2, -1)
    selected_grid = grid[mask].view(2, -1)

    sampling_map[:, selected_locations[1], selected_locations[0]] = selected_grid

    # normalize
    sampling_map[0, :, :] = (sampling_map[0, :, :] / width) * 2 - 1
    sampling_map[1, :, :] = (sampling_map[1, :, :] / height) * 2 - 1

    return sampling_map.reshape((2, height, width))


def tensor_warping(input_image, sampling_map, mode='bilinear'):
    return functional.grid_sample(
        input_image,
        sampling_map,
        mode=mode,
        padding_mode='zeros',
        align_corners=True,
    )


class BackReprojection(nn.Module):
    @staticmethod
    def create_loc_matrix(depth_value):
        device = depth_value.device
        height, width = depth_value.shape[-2:]

        x = torch.linspace(start=0.0, end=width-1, steps=width, device=device)
        y = torch.linspace(start=0.0, end=height-1, steps=height, device=device)

        # Create H x W grids
        y, x = torch.meshgrid(y, x)
        
        x = x.view(1, height, width)
        y = y.view(1, height, width)
        z = depth_value.view(1, height, width)
        o = torch.ones_like(y)

        return torch.cat([x * z, y * z, z, o], dim=0).view((4, -1))

    def forward(self, src_depth, src_intrinsic, src_extrinsic, dst_intrinsic, dst_extrinsic):
        height, width = src_depth.shape[-2:]

        pos_matrix = self.compute_pos_matrix(src_depth, src_intrinsic, src_extrinsic, dst_intrinsic, dst_extrinsic, height, width)

        sampling_map = (pos_matrix[:2, :, :] / (pos_matrix[2:3, :, :] + 1e-7)).reshape((1, 2, height, width))
        sampling_map = sampling_map.permute((0, 2, 3, 1))

        sampling_map[:, :, :, 0:1] = sampling_map[:, :, :, 0:1] / (width / 2) - 1
        sampling_map[:, :, :, 1:2] = sampling_map[:, :, :, 1:2] / (height / 2) - 1

        return sampling_map

    def compute_pos_matrix(self, src_depth, src_intrinsic, src_extrinsic, dst_intrinsic, dst_extrinsic, height, width):
        # compute location matrix
        pos_matrix = BackReprojection.create_loc_matrix(src_depth).reshape(4, -1)
        pos_matrix = torch.linalg.inv((src_intrinsic @ src_extrinsic)) @ pos_matrix
        pos_matrix = dst_intrinsic @ dst_extrinsic @ pos_matrix
        pos_matrix = pos_matrix.reshape((4, height, width))
        return pos_matrix
