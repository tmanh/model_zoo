import math
import copy
import torch
import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from torch.nn.init import normal_

from mmcv.cnn import build_norm_layer, build_plugin_layer, build_upsample_layer, ConvModule
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer, xavier_init
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING

from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_attention, build_transformer_layer, build_feedforward_network, TransformerLayerSequence
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv import ConfigDict


conv_cfg = {'Conv': nn.Conv2d}

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
DEPTHER = MODELS

TRANSFORMER = Registry('Transformer')


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_depther(cfg, train_cfg=None, test_cfg=None):
    """Build depther."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn('train_cfg and test_cfg is deprecated, please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, 'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, 'test_cfg specified in both outer field and model field '
    return DEPTHER.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.
    Args:
        cfg (None or dict): Cfg should contain:
            type (str): Identify conv layer type.
            layer args: Args needed to instantiate a conv layer.
    Returns:
        nn.Module: Created conv layer.
    """

    cfg_ = dict(type='Conv')

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = conv_cfg[layer_type]

    return conv_layer(*args, **kwargs, **cfg_)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), f'when normalize is set, scale should be provided and in float or int type, found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing ignored positions, while zero values means valid positions for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.
    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer. Default: None
        norm_cfg (dict): dictionary to construct and config norm layer. Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer. Default: False
    """

    def __init__(self, block, inplanes, planes, num_blocks, stride=1, dilation=1, avg_down=False, conv_cfg=None, norm_cfg=dict(type='BN'), multi_grid=None, contract_dilation=False, **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
            downsample.extend([build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1]])
            downsample = nn.Sequential(*downsample)

        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]
        layers = [block(inplanes=inplanes, planes=planes, stride=stride, dilation=first_dilation, downsample=downsample, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs)]

        inplanes = planes * block.expansion
        layers.extend(block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation if multi_grid is None else multi_grid[i], conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs) for i in range(1, num_blocks))

        super(ResLayer, self).__init__(*layers)


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.
    This module consists of several plain convolutional layers.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample the input feature map. If stride=2, it only uses stride convolution in the first convolutional layer to downsample the input feature map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the receptive field. Set dilation rate of each convolutional layer and the dilation rate of the first convolutional layer is always 1. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer. Default: None.
        norm_cfg (dict | None): Config dict for normalization layer. Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule. Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not. Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), dcn=None, plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp

        convs = [
            ConvModule(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=3, stride=stride if i == 0 else 1,
                       dilation=1 if i == 0 else dilation, padding=1 if i == 0 else dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg) for i in range(num_convs)]

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""
        return cp.checkpoint(self.convs, x) if self.with_cp and x.requires_grad else self.convs(x)


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.
    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.
    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block. Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer. Default: None.
        norm_cfg (dict | None): Config dict for normalization layer. Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule. Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in decoder. Default: dict(type='InterpConv'). If the size of high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not. Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2,
                 stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), upsample_cfg=dict(type='InterpConv'), dcn=None, plugins=None):
        super(UpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(in_channels=2 * skip_channels, out_channels=out_channels, num_convs=num_convs, stride=stride, dilation=dilation, with_cp=with_cp, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg, dcn=None, plugins=None)
        
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(cfg=upsample_cfg, in_channels=in_channels, out_channels=skip_channels, with_cp=with_cp, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out


class BasicBlock(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None, init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.
    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if it is "caffe", the stride-two layer is the first 1x1 conv layer.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, plugins=None, init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv1']
            self.after_conv2_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv2']
            self.after_conv3_plugins = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv3']

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(conv_cfg, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(dcn, planes, planes, kernel_size=3, stride=self.conv2_stride, padding=dilation, dilation=dilation, bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.
        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.
        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(plugin, in_channels=in_channels, postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        """Forward function for plugins."""
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@TRANSFORMER.register_module()
class PureMSDEnTransformer(BaseModule):
    """
    transformer that only includes an deformable multi-scale encoder
    """
    def __init__(self, encoder=None, init_cfg=None, num_feature_levels=4):
        super(PureMSDEnTransformer, self).__init__(init_cfg=init_cfg)
        self.num_feature_levels = num_feature_levels
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = self.encoder.embed_dims
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """Initialize the transformer weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid points on the feature map, has shape (bs, num_levels, 2)
            device (obj:`device`): The device where reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, mlvl_feats, mlvl_masks, mlvl_pos_embeds, **kwargs):
        """Forward function for `Transformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from different level. Each element has shape [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from different level used for encoder and decoder, each element has shape  [bs, h, w].
            mlvl_pos_embeds (list(Tensor)): The positional encoding of feats from different level, has the shape [bs, embed_dims, h, w].
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor - memory: Encoder results.
        """
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)                     # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)   # (H*W, bs, embed_dims)
        memory = self.encoder(query=feat_flatten, key=None, value=None, query_pos=lvl_pos_embed_flatten, query_key_padding_mask=mask_flatten, spatial_shapes=spatial_shapes,
            reference_points=reference_points, level_start_index=level_start_index, valid_ratios=valid_ratios, **kwargs)

        memory = memory.permute(1, 0, 2)

        return memory


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default： `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in {self.__class__.__name__}, Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER.register_module()
class PixelTransformerDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict` named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`. More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules, The order of the configs in the list should be consistent with
            corresponding attentions in operation_order. If it is a dict, all of the attention modules in operation_order will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be consistent with corresponding ffn in operation_order. If it is a dict, all of the attention modules in operation_order will be built with this config.
        operation_order (tuple[str]): The execution order of operation in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm'). Support `prenorm` when you specifying first element as `norm`. Default：None.
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization. Default: None.
        batch_first (bool): Key, Query and Value are shape of (batch, n, embed_dim) or (n, batch, embed_dim). Default to False.
    """
    def __init__(self, attn_cfgs=None, ffn_cfgs=dict(type='FFN', embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0., act_cfg=dict(type='ReLU', inplace=True)),
                 operation_order=None, norm_cfg=dict(type='LN'), init_cfg=None, batch_first=False, **kwargs):

        deprecated_args = dict(feedforward_channels='feedforward_channels', ffn_dropout='ffn_drop', ffn_num_fcs='num_fcs')

        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(f'The arguments `{ori_name}` in BaseTransformerLayer has been deprecated, now you should set `{new_name}` and other FFN related arguments to a dict named `ffn_cfgs`.')
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(PixelTransformerDecoderLayer, self).__init__(init_cfg)

        self.batch_first = batch_first

        assert (set(operation_order) & {'self_attn', 'norm', 'ffn', 'cross_attn'}) == set(operation_order), f"The operation_order of {self.__class__.__name__} should contains all four operation type {['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length of attn_cfg {num_attn} is not consistent with the number of attention in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])

                # Some custom attentions used as `self_attn` or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index], dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None, attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, num_heads=8, **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape [num_queries, bs, embed_dims] if self.batch_first is False, else [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs, embed_dims] if self.batch_first is False, else [bs, num_keys, embed_dims].
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in calculation of corresponding attention. The length of it should equal to the number of `attention` in `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_queries]. Only used in `self_attn` layer. Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        self.num_heads = num_heads
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in {self.__class__.__name__}')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of attn_masks {len(attn_masks)} must be equal to the number of attention in operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](query, temp_key, temp_value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=query_pos, key_padding_mask=query_key_padding_mask, **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](query, key, value, identity if self.pre_norm else None, query_pos=query_pos, key_pos=key_pos, attn_mask=attn_masks[attn_index], key_padding_mask=key_padding_mask, **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER.register_module()
class PixelTransformerDecoder(BaseModule):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default： `LN`.
    """

    def __init__(self, hidden_dim, post_norm_cfg=dict(type='LN'), return_intermediate=False, transformerlayers=None, num_layers=9, num_feature_levels=3, init_cfg=None,
                 classify=True, class_num=249, operation='%'):

        super(PixelTransformerDecoder, self).__init__(init_cfg)
        if num_layers!=0:
            if isinstance(transformerlayers, dict):
                transformerlayers = [copy.deepcopy(transformerlayers) for _ in range(num_layers)]
            else:
                assert isinstance(transformerlayers, list) and len(transformerlayers) == num_layers
            self.num_layers = num_layers
            self.layers = ModuleList()
            for i in range(num_layers):
                self.layers.append(build_transformer_layer(transformerlayers[i]))
            
            self.embed_dims = self.layers[0].embed_dims
            self.pre_norm = self.layers[0].pre_norm

            if post_norm_cfg is not None:
                self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
            else:
                self.post_norm = None
        else:
            self.layers = ModuleList()

        self.return_intermediate = return_intermediate

        # output FFNs
        self.bins_embed = nn.Linear(hidden_dim, 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.classify = classify
        if self.classify:
            self.class_embed = MLP(hidden_dim, hidden_dim, class_num, 3)

        # normalize
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_feature_levels = num_feature_levels

        self.num_heads = transformerlayers[-1]['attn_cfgs']['num_heads']

        self.hook_identify = torch.nn.Identity()

        self.operation = operation
        assert operation in ['%', '//'], "only support '%' or '//'. No obvious discrepancy between them."

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1) # [300, 2, 256] -> [2, 300, 256]
        
        outputs_class = None
        if self.classify:
            bins_queries = decoder_output[:, 1:, :]
            class_queries = decoder_output[:, 0, :]
            outputs_class = self.class_embed(class_queries)
        else:
            bins_queries = decoder_output

        outputs_bins = self.bins_embed(bins_queries)
        mask_embed = self.mask_embed(bins_queries)
            
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        outputs_mask = self.hook_identify(outputs_mask)

        return outputs_bins, outputs_mask, outputs_class

    def forward(self, ms_feats, query_embed, query_feat, mask_features, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when return_intermediate is `False`, otherwise it has shape [num_layers, num_query, bs, embed_dims].
        """

        # get the multi-scale feats
        src = ms_feats['src']
        pos = ms_feats['pos']

        predictions_bins = []
        predictions_mask = []
        predictions_class = []

        # QxNxC query_embed, output
        query_embed = query_embed
        output = query_feat

        for idx, layer in enumerate(self.layers):
            # // or level_index = idx // self.num_feature_levels
            if self.operation == '%':
                level_index = idx % self.num_feature_levels
            elif self.operation == '//':
                level_index = idx // self.num_feature_levels
            else:
                raise NotImplementedError

            output = layer(output, src[level_index], src[level_index], query_pos=query_embed, key_pos=pos[level_index], attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs)
            outputs_bins, outputs_mask, outputs_class = self.forward_prediction_heads(output, mask_features)

            predictions_bins.append(outputs_bins)
            predictions_mask.append(outputs_mask)
            predictions_class.append(outputs_class)

        return predictions_bins, predictions_mask, predictions_class
