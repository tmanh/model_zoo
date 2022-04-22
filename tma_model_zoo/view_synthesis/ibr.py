import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np

from ..basics.geometry import *

from ..universal.snet import SNetDS2BN_base_8


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = functional.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class IBRNet(nn.Module):
    def __init__(self, args, in_feat_ch=32, n_samples=64, **kwargs):
        super(IBRNet, self).__init__()
        self.args = args
        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = functional.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat, mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = functional.softmax(x, dim=2)  # color blending
        return torch.sum(rgb_in*blending_weights_valid, dim=2)
        # return torch.cat([rgb_out, sigma_out], dim=-1)


class ModifiedIBRNet(nn.Module):
    def __init__(self, in_feat_ch=16, n_samples=4):
        super().__init__()
        self.local_rank = 0
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples

        self.depth_start = 0
        self.depth_end = 15
        self.depth_num = 48

        self.depth_ranges = np.linspace(self.depth_start, self.depth_end, self.depth_num)

        self.enc_net = SNetDS2BN_base_8(in_channels=3)

        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, src_colors, prj_depths, sampling_maps, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, n_channels, height, width = src_colors.shape

        src_colors = src_colors.view(-1, n_channels, height, width)
        src_feats = self.enc_net(src_colors)

        # scaled_colors = functional.interpolate(src_colors, size=(self.sheight, self.swidth), mode='bilinear', align_corners=True)
        # reduced_feats = functional.interpolate(src_feats, size=(self.sheight, self.swidth), mode='bilinear', align_corners=True)

        src_feats = torch.cat([src_colors, src_feats], dim=1).view(n_samples, n_views, -1, height, width)

        sampling_maps, src_masks = self.compute_sampling_maps(n_samples, n_views, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)
        src_masks = src_masks.view(n_samples, n_views, -1, height, width).permute(2, 0, 3, 4, 1).view(self.depth_num, -1, n_views, 1)

        warped_imgs_srcs = self.warp_images(src_feats, sampling_maps, n_views).permute(2, 0, 1, 3, 4, 5)  # [N, V, D, C, H, W]
        warped_imgs_srcs = warped_imgs_srcs.permute(2, 0, 4, 5, 1, 3).view(self.depth_num, -1, n_views, 16 + 3)

        valid_masks = (prj_depths > 0).float()
        valid_mask = (torch.sum(valid_masks, dim=1) > 0).float()

        out_list = []
        length = n_samples * height * width
        for i in range(length // self.n_samples):
            start = i * self.n_samples
            end = min((i + 1) * self.n_samples, length)
            out_list.append(self.compute(warped_imgs_srcs[:, start:end], src_masks[:, start:end]))
        out_color = torch.cat(out_list, dim=1)

        print(warped_imgs_srcs.shape, src_masks.shape, out_color.shape)
        exit()

        return {'refine': aggregated_img, 'deep_dst_color': None, 'deep_prj_colors': None, 'prj_colors': None, 'dst_color': None, 'deep_src_depths': None,
                'valid_mask': valid_mask}

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

    def warp_images(self, src_images, sampling_maps, n_views):
        warped_imgs_srcs = []
        for d in range(self.depth_num):
            image_list = []
            for view in range(n_views):
                sampling_map = sampling_maps[:, view, d, :, :, :]
                warped_view_image = tensor_warping(src_images[:, view, ...], sampling_map)
                image_list.append(warped_view_image)

            stacked_images = torch.stack(image_list, dim=0)  # src_features: [V, N, C, H, W]
            warped_imgs_srcs.append(stacked_images)
        return torch.stack(warped_imgs_srcs, dim=1)  # [N, D, 1, H, W]

    def compute(self, rgb_feat, mask):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''
        num_views = rgb_feat.shape[2]
        rgb_in = rgb_feat[..., :3]

        weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = functional.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat, mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        # sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = functional.softmax(x, dim=2)  # color blending
        return torch.sum(rgb_in*blending_weights_valid, dim=2)
        # return torch.cat([rgb_out, sigma_out], dim=-1)
