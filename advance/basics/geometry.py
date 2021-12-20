import numpy as np

import torch
import torch.nn.functional as functional


def create_loc_matrix(depth_value, y_start, x_start, height, width):
    y = torch.arange(start=int(y_start), end=int(height+y_start)).view(1, height, 1).repeat(1, 1, width)       # columns
    x = torch.arange(start=int(x_start), end=int(width+x_start)).view(1, 1, width).repeat(1, height, 1)        # rows
    ones = torch.ones((1, height, width))
    z = depth_value * ones

    return torch.cat([x * z, y * z, z, ones], dim=0).view((4, -1))


def create_sampling_map_target2source(depth_value, y_dst, x_dst, y_src, x_src, height, width, dst_intrinsic, dst_extrinsic, src_intrinsic, src_extrinsic):
    # compute location matrix
    pos_matrix = create_loc_matrix(depth_value, y_dst, x_dst, height, width).reshape(4, -1)
    pos_matrix = torch.linalg.inv((dst_intrinsic @ dst_extrinsic)) @ pos_matrix
    pos_matrix = src_intrinsic @ src_extrinsic @ pos_matrix
    pos_matrix = pos_matrix.reshape((4, height, width))

    # compute sampling maps
    sampling_map = pos_matrix[:2, :, :] / (pos_matrix[2:3, :, :] + 1e-7)
    sampling_map[0, :, :] -= x_src
    sampling_map[1, :, :] -= y_src

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


def tensor_warping(input_image, sampling_map):
    return functional.grid_sample(
        input_image,
        sampling_map,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
