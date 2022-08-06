# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.utils import _pair as to_2tuple


from mmcv.cnn import build_norm_layer, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.runner import _load_checkpoint
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import Config

from .depthformer_basics import BACKBONES, ResLayer, BasicBlock, Bottleneck, build_conv_layer
from .depthformer_utils import swin_convert, get_root_logger, resize


def build_backbone_from(path):
    cfg = Config.fromfile(path)
    return BACKBONES.build(cfg.model.backbone)


# Modified from Pytorch-Image-Models
class PatchEmbed(BaseModule):
    """Image to Patch Embedding V2.
    We use a conv layer to implement PatchEmbed.
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (dict, optional): The config dict for conv layers type selection. Default: None.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv. Default: None (Default to be equal with kernel_size).
        padding (int): The padding length of embedding conv. Default: 0.
        dilation (int): The dilation rate of embedding conv. Default: 1.
        pad_to_patch_size (bool, optional): Whether to pad feature map shape to multiple patch size. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
    """

    def __init__(self, in_channels=3, embed_dims=768, conv_type=None, kernel_size=16, stride=16, padding=0, dilation=1, pad_to_patch_size=True, norm_cfg=None):
        super().__init__()

        self.embed_dims = embed_dims

        if stride is None:
            stride = kernel_size

        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, f'The size of patch should have length 1 or 2, but got {len(patch_size)}'

        self.patch_size = patch_size

        # Use conv layer to embed
        conv_type = conv_type or 'Conv2d'
        self.projection = build_conv_layer(dict(type=conv_type), in_channels=in_channels, out_channels=embed_dims, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        self.norm = None
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))

        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer use nn.Unfold to group feature map by kernel_size, and use norm
    and linear layer to embed grouped feature map.
    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        stride (int | tuple): the stride of the sliding length in the unfold layer. Defaults: 2. (Default to be equal with kernel_size).
        bias (bool, optional): Whether to add bias in linear layer or not. Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer. Defaults: dict(type='LN').
    """

    def __init__(self, in_channels, out_channels, stride=2, bias=False, norm_cfg=dict(type='LN')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.sampler = nn.Unfold(kernel_size=stride, dilation=1, padding=0, stride=stride)

        sample_dim = stride**2 * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, hw_shape):
        """
        x: x.shape -> [B, H*W, C]
        hw_shape: (H, W)
        """
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        # stride is fixed to be equal to kernel_size.
        if (H % self.stride != 0) or (W % self.stride != 0):
            x = F.pad(x, (0, W % self.stride, 0, H % self.stride))

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        x = self.sampler(x)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        down_hw_shape = (H + 1) // 2, (W + 1) // 2
        return x, down_hw_shape


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_init(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows, Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shift Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight. Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output. Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output. Defaults: dict(type='DropPath', drop_prob=0.).
    """

    def __init__(self, embed_dims, num_heads, window_size, shift_size=0, qkv_bias=True, qk_scale=None, attn_drop_rate=0, proj_drop_rate=0, dropout_layer=dict(type='DropPath', drop_prob=0.)):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=to_2tuple(window_size), qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)
        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)

        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window size (int, optional): The local window scale. Default: 7.
        shift (bool): whether to shift window or not. Default False.
        qkv_bias (int, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.2.
        act_cfg (dict, optional): The config dict of activation function. Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of nomalization. Default: dict(type='LN').
    """

    def __init__(self, embed_dims, num_heads, feedforward_channels, window_size=7, shift=False, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN')):

        super(SwinBlock, self).__init__()

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2 if shift else 0, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=2, ffn_drop=drop_rate, dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg, add_identity=True)

    def forward(self, x, hw_shape):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window size (int): The local window scale. Default: 7.
        qkv_bias (int): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.2.
        downsample (BaseModule | None, optional): The downsample operation module. Default: None.
        act_cfg (dict, optional): The config dict of activation function. Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of nomalization. Default: dict(type='LN').
    """

    def __init__(self, embed_dims, num_heads, feedforward_channels, depth, window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 downsample=None, act_cfg=dict(type='GELU'), norm_cfg=dict(type='LN')):
        super().__init__()

        drop_path_rate = drop_path_rate if isinstance(
            drop_path_rate,
            list) else [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=feedforward_channels, window_size=window_size, shift=i % 2 != 0,
                              qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate[i], act_cfg=act_cfg, norm_cfg=norm_cfg)

            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@BACKBONES.register_module()
class DepthFormerSwin(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows` - https://arxiv.org/abs/2103.14030

    Inspiration from https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when pretrain. Defaults: 224.
        in_channels (int): The num of input channels. Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage. Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of each Swin Transformer stage. (In swin, we set kernel size equal to stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages. Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer. Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at output of backone. Defaults: dict(type='LN').
        pretrain_style (str): Choose to use official or mmcls pretrain weights. Default: official.
        pretrained (str, optional): model pretrained path. Default: None.
        
        ### Conv cfg
        conv_cfg (dict | None): Dictionary to construct and config conv layer. When conv_cfg is None, cfg will be set to dict(type='Conv2d'). Default: None.
        conv_norm_cfg (dict): Dictionary to construct and config norm layer. Default: None.
        resnet_depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 0. Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some memory while slowing down the training speed. Default: False.
        conv_strides (Sequence[int]): Strides of the first block of each stage. Default: (1, 2, 2, 2).,
        conv_dilations (Sequence[int]): Dilation of each stage. Default: (1, 1, 1, 1).
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'. (No use in this version. We utilize scratched Resnet branch in our experiments)
        conv_pretrained (str, optional): model pretrained path. Default: None. (No use in this version. We utilize scratched Resnet branch in our experiments)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, pretrain_img_size=224, in_channels=3, embed_dims=96, patch_size=4, window_size=7, mlp_ratio=4, depths=None,
                 num_heads=None, strides=None, out_indices=None, qkv_bias=True, qk_scale=None, patch_norm=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, use_abs_pos_embed=False, act_cfg=None, norm_cfg=None, pretrain_style='official',
                 pretrained=None, conv_cfg=None, conv_norm_cfg=None, resnet_depth=None, num_stages=None, conv_strides=None,
                 conv_dilations=None, conv_pretrained=None):
        super(DepthFormerSwin, self).__init__()

        self.norm_cfg = dict(type='LN') if norm_cfg is None else norm_cfg
        self.act_cfg = dict(type='GELU') if act_cfg is None else act_cfg
        
        self.patch_norm_cfg = self.norm_cfg if patch_norm else None

        self.depths = (2, 2, 6, 2) if depths is None else depths
        self.strides = (4, 2, 2, 2) if strides is None else strides

        self.out_indices = (0, 1, 2, 3) if out_indices is None else out_indices
        self.num_heads = (3, 6, 12, 24) if num_heads is None else num_heads
        self.conv_strides = (1, 2, 2, 2) if conv_strides is None else conv_strides
        self.conv_dilations = (1, 1, 1, 1) if conv_dilations is None else conv_dilations

        self.list_feats = []
        self.conv_cfg = conv_cfg
        self.conv_norm_cfg = conv_norm_cfg
        self.use_abs_pos_embed = use_abs_pos_embed
        self.pretrain_style = pretrain_style
        self.pretrained = pretrained
        self.conv_pretrained = conv_pretrained

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        assert pretrain_style in ['official', 'mmcls'], 'We only support load official ckpt and mmcls ckpt.'
        assert self.strides[0] == patch_size, 'Use non-overlapping patch embed.'

        num_layers = len(self.depths)

        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dims=embed_dims, conv_type='Conv2d', kernel_size=patch_size,
                                      stride=self.strides[0], pad_to_patch_size=True, norm_cfg=self.patch_norm_cfg)

        self.init_abs_pos_embed(pretrain_img_size, embed_dims, patch_size)
        self.init_swin_blocks(embed_dims, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, num_layers)
        self.init_norm_layers(embed_dims, num_layers)
        self.init_stem_layer(3, num_stages, resnet_depth)
        self.init_weights()

    def init_norm_layers(self, embed_dims, num_layers):
        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        
        # Add a norm layer for each output
        for i in self.out_indices:
            layer = build_norm_layer(self.norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def init_swin_blocks(self, embed_dims, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, num_layers):
        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(in_channels=in_channels, out_channels=2 * in_channels, stride=self.strides[i + 1], norm_cfg=self.patch_norm_cfg)
            else:
                downsample = None

            stage = SwinBlockSequence(embed_dims=in_channels, num_heads=self.num_heads[i], feedforward_channels=mlp_ratio * in_channels, depth=self.depths[i], window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[:self.depths[i]], downsample=downsample, act_cfg=self.act_cfg, norm_cfg=self.norm_cfg)
            self.stages.append(stage)

            dpr = dpr[self.depths[i]:]
            if downsample:
                in_channels = downsample.out_channels

    def init_abs_pos_embed(self, pretrain_img_size, embed_dims, patch_size):
        pretrain_img_size = self.convert2tuple(pretrain_img_size)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(torch.zeros((1, num_patches, embed_dims)))

    def convert2tuple(self, pretrain_img_size):
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, f'The size of image should have length 1 or 2, but got {len(pretrain_img_size)}'

        return pretrain_img_size

    @property
    def _conv_stem_norm1(self):
        return getattr(self, self._conv_stem_norm1_name)

    def init_stem_layer(self, in_channels, num_stages, resnet_depth):
        self.conv1 = build_conv_layer(self.conv_cfg, in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._conv_stem_norm1_name, _conv_stem_norm1 = build_norm_layer(self.conv_norm_cfg, 64, postfix=1)
        self.add_module(self._conv_stem_norm1_name, _conv_stem_norm1)
        self._conv_stem_relu = nn.ReLU(inplace=True)
        self._conv_stem_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # default depth=50, num_stages=1, 2, 3, 4
        self.num_stages = num_stages
        self.inplanes = 64
        if num_stages != 0:
            self.block, stage_blocks = self.arch_settings[resnet_depth]
            self.stage_blocks = stage_blocks[:num_stages]
            self.res_layers = []
            for i, num_blocks in enumerate(self.stage_blocks):
                stride = self.conv_strides[i]
                dilation = self.conv_dilations[i]
                planes = 64 * 2**i
                res_layer = ResLayer(self.block, self.inplanes, planes, num_blocks, stride=stride, dilation=dilation, style=self.style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=conv_norm_cfg)
                self.inplanes = planes * self.block.expansion
                layer_name = f'layer{i + 1}'
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)

    def init_default_weights(self):
        if self.use_abs_pos_embed:
            trunc_normal_init(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)

    def load_checkpoint(self):
        logger = get_root_logger()
        ckpt = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
            
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        return logger, state_dict

    def standardize_state_dict(self, logger, state_dict):
        if self.pretrain_style == 'official':
            state_dict = swin_convert(state_dict)

        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        self.reshape_state_dict_abs_pos_embed(logger, state_dict)
        self.interpolate_pos_bias_table(logger, state_dict)

    def interpolate_pos_bias_table(self, logger, state_dict):
        relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
        for table_key in relative_position_bias_table_keys:
            table_pretrained = state_dict[table_key]
            table_current = self.state_dict()[table_key]
            L1, nH1 = table_pretrained.size()
            L2, nH2 = table_current.size()
            if nH1 != nH2:
                logger.warning(f'Error in loading {table_key}, pass')
            elif L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                table_pretrained_resized = resize(table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()

    def reshape_state_dict_abs_pos_embed(self, logger, state_dict):
        if state_dict.get('absolute_pos_embed') is not None:
            absolute_pos_embed = state_dict['absolute_pos_embed']
            N1, L, C1 = absolute_pos_embed.shape
            N2, C2, H, W = self.absolute_pos_embed.shape
            if N1 != N2 or C1 != C2 or L != H * W:
                logger.warning('Error in loading absolute_pos_embed, pass')
            else:
                state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

    def init_weights(self):
        if self.pretrained is None:
            super().init_weights()
            self.init_default_weights()

        if isinstance(self.pretrained, str):
            logger, state_dict = self.load_checkpoint()
            self.standardize_state_dict(logger=logger, state_dict=state_dict)
            self.load_state_dict(state_dict, False)

    def conv_stem(self, x):
        conv_stem = self.conv1(x)
        conv_stem = self._conv_stem_norm1(conv_stem)
        conv_stem = self._conv_stem_relu(conv_stem)

        if self.num_stages != 0:
            x = self._conv_stem_maxpool(x)
            for layer_name in self.res_layers:
                res_layer = getattr(self, layer_name)
                conv_stem = res_layer(conv_stem)

        return conv_stem

    def forward(self, x):
        conv_stem = self.conv_stem(x)
        outs = [conv_stem]
        x = self.patch_embed(x)

        hw_shape = (self.patch_embed.DH, self.patch_embed.DW)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs
