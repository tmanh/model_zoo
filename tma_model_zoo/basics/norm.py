import torch.nn as nn


class NormBuilder:
    NORM_LAYERS = {'BN1d': nn.BatchNorm1d, 'BN2d': nn.BatchNorm2d, 'BN3d': nn.BatchNorm3d,
                   'SyncBN': nn.SyncBatchNorm, 'GN': nn.GroupNorm, 'LN': nn.LayerNorm,
                   'IN1d': nn.InstanceNorm1d, 'IN2d': nn.InstanceNorm2d, 'IN3d': nn.InstanceNorm3d}

    @staticmethod
    def build(cfg, num_features, postfix=''):
        """Build normalization layer.

        Args:
            cfg (dict): The norm layer config, which should contain:

                - type (str): Layer type.
                - layer args: Args needed to instantiate a norm layer.
                - requires_grad (bool, optional): Whether stop gradient updates.
            num_features (int): Number of input channels.
            postfix (int | str): The postfix to be appended into norm abbreviation
                to create named layer.

        Returns:
            (str, nn.Module): The first element is the layer name consisting of
                abbreviation and postfix, e.g., bn1, gn. The second element is the
                created norm layer.
        """
        NormBuilder.raise_error(cfg)
        cfg_ = cfg.copy()

        layer_type = cfg_.pop('type')
        if layer_type not in NormBuilder.NORM_LAYERS:
            raise KeyError(f'Unrecognized norm type {layer_type}')

        norm_layer = NormBuilder.NORM_LAYERS[layer_type]

        requires_grad = cfg_.pop('requires_grad', True)
        cfg_.setdefault('eps', 1e-5)

        if layer_type != 'GN':
            layer = norm_layer(num_features, **cfg_)
            if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
                layer._specify_ddp_gpu_num(1)
        else:
            assert 'num_groups' in cfg_
            layer = norm_layer(num_channels=num_features, **cfg_)

        if not requires_grad:
            if hasattr(layer, 'momentum'):
                layer.momentum = 0
            layer.eval()

        for param in layer.parameters():
            param.requires_grad = requires_grad

        return layer

    @staticmethod
    def raise_error(cfg):
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
