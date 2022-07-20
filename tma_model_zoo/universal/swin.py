from importlib.metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils.checkpoint as checkpoint

import numpy as np

from ..basics.dynamic_conv import DynamicConv2d
from ..basics.norm import NormBuilder
from .drop import DropPath
from .weight_init import trunc_normal_
from .helpers import to_2tuple


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., requires_grad=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        for param in self.fc1.parameters():
            param.requires_grad = requires_grad
        for param in self.fc2.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_partitionA(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_reverseA(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)



class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., requires_grad=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=requires_grad)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table of the shifting window in range (-h, h) and (-w, w)
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))                 # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                                  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1                        # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1                        # shift to start from 0
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        for param in self.cpb_mlp.parameters():
            param.requires_grad = requires_grad
        for param in self.qkv.parameters():
            param.requires_grad = requires_grad
        for param in self.proj.parameters():
            param.requires_grad = requires_grad
        self.q_bias.requires_grad = requires_grad
        self.v_bias.requires_grad = requires_grad

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # Idea: The model generate 3 vectors, Q and K are 2 vectors to compute the attention (kind of attention of attention). V is the values/feats vector
        # Step 1: generate Q, K, V vectors
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))  # Note: Q and V has bias, but not K
        qkv = functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)                                   
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                                                          # make torchscript happy (cannot use tensor as tuple)

        # Step 2: cosine attention - joint attention is computed from Q and K
        attn = (functional.normalize(q, dim=-1) @ functional.normalize(k, dim=-1).transpose(-2, -1))              
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        # Step 3: Generate attention bias based on the relative location of pixel in the window
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Step 4: added mask to the attention (no attention for missing/hole pixels)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Step 5: normalize the attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Step 6: Apply attention on the V vector, then refine the feats
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class WindowAttentionA(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, requires_grad=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.q_conv = DynamicConv2d(dim, dim, kernel_size=1, norm_cfg=None, act=None, bias=True, requires_grad=requires_grad)
        self.k_conv = DynamicConv2d(dim, dim, kernel_size=1, norm_cfg=None, act=None, requires_grad=requires_grad)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        for param in self.q.parameters():
            param.requires_grad = requires_grad
        for param in self.k.parameters():
            param.requires_grad = requires_grad
        for param in self.v.parameters():
            param.requires_grad = requires_grad
        for param in self.proj.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, C, K, K = x.shape

        # Idea: The model generate 3 vectors, Q and K are 2 vectors to compute the attention (kind of attention of attention). V is the values/feats vector
        # Step 1: generate Q, K, V vectors
        q = self.q(x).view(B, C, -1).permute(0, 2, 1)

        k = torch.max_pool2d(x, kernel_size=2, stride=2)
        k = self.k(k).view(B, C, -1)
        
        qk = q @ k
        qk = torch.max(qk, dim=2, keepdims=True)[0]

        v = torch.mean(x, dim=(2, 3), dtype=None, keepdims=True).view(B, C, -1).permute(0, 2, 1)

        # Step 2: Apply attention on the V vector, then refine the feats
        out = (qk @ v).permute(0, 2, 1).view(B, C, K, K) 
        out = self.proj(out)
        return out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer='LN', requires_grad=True):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, requires_grad=requires_grad)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, requires_grad=requires_grad)

        self.input_resolution = None
        self.register_buffer("attn_mask", None)

    def compute_attn_mask(self, input_resolution, device):
        if self.input_resolution is not None and self.input_resolution == input_resolution:
            return

        self.input_resolution = input_resolution

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, resolution):
        H, W = resolution
        B, _, C = x.shape
        assert self.window_size ** 2 <= H * W, f'Window size {self.window_size} is too big for input size {resolution}'

        shortcut = x
        x = x.view(B, H, W, C)

        # Step 0: pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        # Step 0: compute the attention mask
        self.compute_attn_mask((H_pad, W_pad), device=x.device)

        # Step 1: compute shift offsets if cyclic shift is needed
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Step 2: partition windows (extracting the non-overlapping windows)
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Step 3: W-MSA/SW-MSA (computing the "attended" feats for each windows)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # Step 4: merge windows (merge the non-overlapping windows back to 1 single image)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # B H' W' C

        # Step 5: reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # Step 6: FFN - residual refinement on the features
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class MixFFN(nn.Module):
    def __init__(self, dim, norm_layer, ratio, requires_grad):
        super().__init__()

        expand_dim = int(dim * ratio)

        self.conv1 = nn.Conv2d(dim, expand_dim, kernel_size=1, bias=False)
        self.bn1 =  NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=expand_dim)

        self.conv2 = nn.Conv2d(expand_dim, expand_dim, kernel_size=3, padding=1, groups=expand_dim, bias=False)
        self.bn2 =  NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=expand_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(expand_dim, dim, kernel_size=1, bias=False)

        for param in self.conv1.parameters():
            param.requires_grad = requires_grad
        for param in self.conv2.parameters():
            param.requires_grad = requires_grad
        for param in self.conv3.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        return self.conv3(x2)


class SwinTransformerBlockA(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer='BN2d', requires_grad=True):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=dim)
        self.attn = WindowAttentionA(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, requires_grad=requires_grad)

        self.norm2 = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=dim)
        
        self.ffn = MixFFN(dim, norm_layer, mlp_ratio, requires_grad=requires_grad)

        self.input_resolution = None
        self.register_buffer("attn_mask", None)

    def compute_attn_mask(self, input_resolution, device):
        if self.attn_mask is not None and (self.input_resolution is not None and self.input_resolution == input_resolution):
            return
        self.input_resolution = input_resolution

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, resolution):
        H, W = resolution
        assert self.window_size ** 2 <= H * W, f'Window size {self.window_size} is too big for input size {resolution}'
        shortcut = x

        # Step 0: pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = functional.pad(x, (0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[2], x.shape[3]

        # Step 0: compute the attention mask
        self.compute_attn_mask((H_pad, W_pad), device=x.device)

        # Step 1: compute shift offsets if cyclic shift is needed
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # Step 2: partition windows (extracting the non-overlapping windows)
        x_windows = window_partitionA(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        
        # Step 3: W-MSA/SW-MSA (computing the "attended" feats for each windows)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # Step 4: merge windows (merge the non-overlapping windows back to 1 single image)
        shifted_x = window_reverseA(attn_windows, self.window_size, H_pad, W_pad)  # B H' W' C

        # Step 5: reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        x = shortcut + self.norm1(x)

        # Step 6: FFN - residual refinement on the features
        x = x + self.ffn(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    """ Patch Merging Layer (or downsampling).
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, stride=2, norm_layer='LN', requires_grad=True):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=2 * dim)

        for param in self.reduction.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, resolution):
        """
        x: B, H*W, C
        """
        H, W = resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Step 1: change h * w ==> h x w
        x = x.view(B, H, W, C).permute([0, 3, 1, 2])

        # stride is fixed to be equal to kernel_size.
        if (H % self.stride != 0) or (W % self.stride != 0):
            x = functional.pad(x, (0, W % self.stride, 0, H % self.stride))

        # Step 2: Shuffle to merge the features from 4 neighbor pixels
        x = x.permute([0, 2, 3, 1])
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        resolution = x.shape[1:3]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # Step 3: refine the feats
        if isinstance(self.norm, nn.BatchNorm1d):
            return self.norm(self.reduction(x).permute(0, 2, 1)).permute(0, 2, 1), resolution
        return self.norm(self.reduction(x)), resolution

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class PatchMergingA(nn.Module):
    """ Patch Merging Layer (or downsampling).
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, stride=2, norm_layer='BN2d', requires_grad=True):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1, bias=False)
        self.norm = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=2 * dim)

        for param in self.reduction.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, resolution):
        """
        x: B, H*W, C
        """
        H, W = resolution
        B, _, _, C = x.shape

        # Step 1: stride is fixed to be equal to kernel_size.
        if (H % self.stride != 0) or (W % self.stride != 0):
            x = functional.pad(x, (0, W % self.stride, 0, H % self.stride))

        # Step 2: Shuffle to merge the features from 4 neighbor pixels
        x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], dim=1)  # B H/2 W/2 4*C

        resolution = x.shape[-2:]

        # Step 3: refine the feats
        return self.norm(self.reduction(x)), resolution

    def extra_repr(self) -> str:
        return f"dim={self.dim}"



class SwinTransformerLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer='LN', downsample=None, requires_grad=True):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, requires_grad=requires_grad)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, requires_grad=requires_grad)
        else:
            self.downsample = None

    def forward(self, x, resolution):
        for blk in self.blocks:
            x = blk(x, resolution)
        
        x_down = x
        resolution_down = resolution
        if self.downsample is not None:
            x_down, resolution_down = self.downsample(x_down, resolution_down)
        return x, resolution, x_down, resolution_down

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinTransformerLayerA(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, norm_layer='LN', downsample=None, requires_grad=True):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlockA(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, requires_grad=requires_grad) for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, requires_grad=requires_grad)
        else:
            self.downsample = None

    def forward(self, x, resolution):
        for blk in self.blocks:
            x = blk(x, resolution)
        
        x_down = x
        resolution_down = resolution
        if self.downsample is not None:
            x_down, resolution_down = self.downsample(x_down, resolution_down)
        return x, resolution, x_down, resolution_down

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbedA(nn.Module):
    """ Image to Patch Embedding (extract feats and flatten the feature map)
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None, image_size=None, requires_grad=True):
        super().__init__()
        if image_size:
            self.compute_patch_stats(image_size)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj0 = DynamicConv2d(in_chans, embed_dim[0], kernel_size=3, stride=1, norm_cfg='BN2d')
        self.proj1 = DynamicConv2d(embed_dim[0], embed_dim[1], kernel_size=2, stride=2, norm_cfg='BN2d')
        self.proj2 = nn.Conv2d(embed_dim[1], embed_dim[2], kernel_size=2, stride=2)
        self.norm = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=embed_dim[2]) if norm_layer is not None else None
        
        for param in self.proj0.parameters():
            param.requires_grad = requires_grad
        for param in self.proj1.parameters():
            param.requires_grad = requires_grad
        for param in self.proj2.parameters():
            param.requires_grad = requires_grad

    def compute_patch_stats(self, img_size):
        img_size = to_2tuple(img_size)
        patches_resolution = [img_size[0] // 4, img_size[1] // 4]
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

    def forward(self, x):
        height, width = x.shape[-2:]

        x0 = self.proj0(x)
        x1 = self.proj1(x0)
        x2 = self.proj2(x1)  # B Ph*Pw C

        x2 = self.norm(x2)

        h0, w0 = x0.shape[-2:]
        h1, w1 = x1.shape[-2:]
        return [x0, x1, x2], [(h0, w0), (h1, w1), (height // 4, width // 4)]


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding (extract feats and flatten the feature map)
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, image_size=None, requires_grad=True):
        super().__init__()
        if image_size:
            self.compute_patch_stats(image_size)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = NormBuilder.build(cfg=dict(type=norm_layer, requires_grad=requires_grad), num_features=embed_dim) if norm_layer is not None else None
        
        for param in self.proj.parameters():
            param.requires_grad = requires_grad

    def compute_patch_stats(self, img_size):
        img_size = to_2tuple(img_size)
        patches_resolution = [img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]]
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

    def forward(self, x):
        height, width = x.shape[-2:]
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, [height // self.patch_size[0], width // self.patch_size[1]]


class SwinTransformerV2(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows` - https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        img_size (int | tuple(int)): Input image size. Default None
    """

    pretrain = '/scratch/antruong/workspace/myspace/model_zoo/pretrained/swinv2_tiny_patch4_window8_256.pth'

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, depths = None, num_heads = None, window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, ape=False,
                 norm_layer='LN', img_size=None, requires_grad=True, **kwargs):
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        
        super().__init__()

        self.list_feats = [96, 192, 384, 768]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer, image_size=img_size, requires_grad=requires_grad)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

            self.absolute_pos_embed.requires_grad = requires_grad

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer,
                                         downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, requires_grad=requires_grad)
            self.layers.append(layer)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x):
        x, resolution = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outputs = []
        resolutions = []
        for layer in self.layers:
            x_before_downscale, resolution_before_downscale, x, resolution = layer(x, resolution=resolution)

            outputs.append(x_before_downscale)
            resolutions.append(resolution_before_downscale)

        return outputs, resolutions

    def load_pretrained(self):
        if isinstance(self.pretrain, str):
            state_dict = torch.load(self.pretrain)

        self = self.load_state_dict(state_dict['model'], strict=False)


class SwinTransformerA(nn.Module):
    """ Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows` - https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        img_size (int | tuple(int)): Input image size. Default None
    """

    pretrain = '/scratch/antruong/workspace/myspace/model_zoo/pretrained/swinv2_tiny_patch4_window8_256.pth'

    def __init__(self, in_chans=3, embed_dim=96, depths = None, num_heads = None, window_size=8, mlp_ratio=4., qkv_bias=True, norm_layer='BN2d', img_size=None, requires_grad=True):  # BN1d, LN
        if depths is None:
            depths = [3, 10, 16, 5]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        
        super().__init__()

        self.list_feats = [32, 48, 64, 128, 256, 512]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedA(in_chans=in_chans, embed_dim=self.list_feats[:3], norm_layer=norm_layer, image_size=img_size, requires_grad=requires_grad)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerLayerA(dim=int(self.list_feats[i_layer + 2]), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias, norm_layer=norm_layer, downsample=PatchMergingA if (i_layer < self.num_layers - 1) else None, requires_grad=requires_grad)
            self.layers.append(layer)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x):
        embed, resolution = self.patch_embed(x)

        x0, x1, x = embed
        r0, r1, resolution = resolution

        outputs = [x0, x1]
        resolutions = [r0, r1]
        for layer in self.layers:
            x_before_downscale, resolution_before_downscale, x, resolution = layer(x, resolution=resolution)

            outputs.append(x_before_downscale)
            resolutions.append(resolution_before_downscale)

        return outputs, resolutions

    def load_pretrained(self):
        if isinstance(self.pretrain, str):
            state_dict = torch.load(self.pretrain)

        self = self.load_state_dict(state_dict['model'], strict=False)
