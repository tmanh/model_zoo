# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_

import math
import torch

from ..basics.dynamic_conv import DynamicConv2d

from .weight_init import xavier_init
from .positional_encoding import SinePositionalEncoding

from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention


# position embedding for fusion layer
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()


class HAHIHetero(nn.Module):
    """HAHIHetero.

    For heterogenenious cnn- and transformer-features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map. Default: [1, 1, 1, 1]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule. Default: None.
    """

    def __init__(self, in_channels, out_channels, embedding_dim, scales = None, norm_cfg='BN2d', act=nn.ReLU(inplace=True), cross_att=True, self_att=True, constrain=False,
                 positional_encoding=SinePositionalEncoding, num_points=8, requires_grad=True, num_feature_levels=6):
        if scales is None:
            scales = [1 for _ in range(num_feature_levels)]
        
        super().__init__()
        assert isinstance(in_channels, list)
        self.cross_att = cross_att
        self.self_att = self_att
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.embedding_dim = embedding_dim
        self.constrain = constrain

        self.lateral_convs = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.lateral_convs.append(DynamicConv2d(in_channel, out_channel, kernel_size=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad))

        self.trans_proj = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels[1:], out_channels[1:]):
            self.trans_proj.append(DynamicConv2d(out_channel, self.embedding_dim, kernel_size=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad))

        self.trans_fusion = nn.ModuleList()
        for in_channel, out_channel in zip(out_channels[1:], out_channels[1:]):
            self.trans_fusion.append(DynamicConv2d(out_channel + self.embedding_dim, out_channel, kernel_size=3, stride=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad))

        self.conv_proj = DynamicConv2d(in_channels[0], self.embedding_dim, kernel_size=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)
        self.conv_fusion = DynamicConv2d(in_channels[0] + self.embedding_dim, out_channels[0], kernel_size=3, stride=1, norm_cfg=norm_cfg, act=act, requires_grad=requires_grad)

        ########################################

        num_feature_levels = num_feature_levels # transformer feature level

        self.trans_positional_encoding = positional_encoding(num_feats=128)
        self.conv_positional_encoding = positional_encoding(num_feats=128)

        self.reference_points = nn.Linear(self.embedding_dim, 2)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, self.embedding_dim))

        self.multi_att = MultiScaleDeformableAttention(embed_dims=self.embedding_dim, num_levels=num_feature_levels-1, num_heads=8, num_points=num_points, batch_first=True)
        self.self_attn = MultiScaleDeformableAttention(embed_dims=self.embedding_dim, num_levels=num_feature_levels-1, num_heads=8, num_points=num_points, batch_first=True)

        for param in self.trans_positional_encoding.parameters():
            param.requires_grad = requires_grad
        for param in self.conv_positional_encoding.parameters():
            param.requires_grad = requires_grad
        for param in self.reference_points.parameters():
            param.requires_grad = requires_grad
        for param in self.multi_att.parameters():
            param.requires_grad = requires_grad
        for param in self.self_attn.parameters():
            param.requires_grad = requires_grad
        self.level_embed.requires_grad = requires_grad

        if not requires_grad:
            self.trans_positional_encoding.eval()
            self.conv_positional_encoding.eval()
            self.multi_att.eval()
            self.self_attn.eval()

        self.init_weights()
        
    # init weight
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
            
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # input projection
        feats_projed = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        feats_trans = feats_projed[1:]
        feat_conv = feats_projed[0]

        src, fusion_res_conv = self.self_attention(feats_trans, feat_conv)

        # unfold the feats back to the origin shape
        start = 0
        fusion_res_trans = []
        for i in range(len(feats_trans)):
            bs, c, h, w = feats_trans[i].shape
            end = start + h * w
            feat = src[:, start:end, :].permute(0, 2, 1).contiguous()
            start = end
            feat = feat.reshape(bs, self.embedding_dim, h, w)
            fusion_res_trans.append(torch.cat([feats_trans[i], feat], dim=1))

        # fusion 3x3 conv
        outs = []
        for i in range(len(feats_trans)):
            x_resize = self.trans_fusion[i](fusion_res_trans[i])
            outs.append(x_resize)
        outs.insert(0, fusion_res_conv)

        return tuple(outs)

    def self_attention(self, feats_trans, feat_conv):
        # HI (deformable self attention)
        masks = []
        src_flattens = []
        spatial_shapes = []
        lvl_pos_embed_flatten = []
        for i in range(len(feats_trans)):
            bs, c, h, w = feats_trans[i].shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            mask = torch.zeros_like(feats_trans[i][:, 0, :, :]).bool()
            masks.append(mask)

            pos = self.trans_positional_encoding(mask)
            pos_embed = pos.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[i].view(1, 1, -1)

            feat = self.trans_proj[i](feats_trans[i])
            flatten_feat = feat.flatten(2).transpose(1, 2)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flattens.append(flatten_feat)

        src_flatten = torch.cat(src_flattens, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
        if self.self_att:
            src = self.self_attn(src_flatten, key=None, value=None,
                identity=None,
                query_pos=lvl_pos_embed_flatten,
                key_padding_mask=None,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,)
        else:
            src = src_flatten

        # HA (deformable cross attention)
        conv_skip = self.conv_proj(feat_conv)
        bs, c, h, w = conv_skip.shape
        query_mask = torch.zeros_like(conv_skip[:, 0, :, :]).type(torch.bool)
        query = conv_skip.flatten(2).transpose(1, 2)
        query_embed = self.conv_positional_encoding(query_mask).flatten(2).transpose(1, 2)
        reference_points = self.reference_points(query_embed).sigmoid()
        reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

        if self.cross_att:
            fusion_res_conv = self.multi_att(query, key=None, value=src, identity=None,
                query_pos=query_embed,
                key_padding_mask=None,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,)
        else:
            fusion_res_conv = query

        fusion_res_conv = fusion_res_conv.permute(0, 2, 1).reshape(bs, c, h, w)
        fusion_res_conv = self.conv_fusion(torch.cat([fusion_res_conv, feat_conv], dim=1))
        return src, fusion_res_conv
