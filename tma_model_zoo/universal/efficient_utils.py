from opcode import stack_effect
import re
import math
import collections

import torch
from torch.utils import model_zoo


# train with Standard methods
# check more details in paper(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)
url_map = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

# train with Adversarial Examples(AdvProp)
# check more details in paper(Adversarial Examples Improve Image Recognition)
url_map_advprop = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',
    'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',
}


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top', 'need_norm', 'gating'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Set GlobalParams and BlockArgs's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    return inputs / keep_prob * binary_tensor


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.
    Args:
        model_name (str): Model name to be queried.
    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.
        Args:
            block (namedtuple): A BlockArgs type argument.
        Returns:
            block_string: A String form of BlockArgs.
        """
        args = ['r%d' % block.num_repeat, 'k%d' % block.kernel_size, 's%d%d' % (block.strides[0], block.strides[1]), f'e{block.expand_ratio}', 'i%d' % block.input_filters, 'o%d' % block.output_filters]

        if 0 < block.se_ratio <= 1:
            args.append(f'se{block.se_ratio}')
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        return [
            BlockDecoder._decode_block_string(block_string)
            for block_string in string_list
        ]

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        return [BlockDecoder._encode_block_string(block) for block in blocks_args]


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True):
    """Create BlockArgs and GlobalParams for efficientnet model.
    Args:
        width_coefficient (float)
        depth_coefficient (float)
        image_size (int)
        dropout_rate (float)
        drop_connect_rate (float)
        num_classes (int)
        Meaning as the name suggests.
    Returns:
        blocks_args, global_params.
    """

    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        dropout_rate=dropout_rate,

        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
        need_norm=True,
        gating=False,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.
    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.
    Returns:
        blocks_args, global_params
    """
    if not model_name.startswith('efficientnet'):
        raise NotImplementedError(f'model name is not pre-defined: {model_name}')

    w, d, s, p = efficientnet_params(model_name)
    # note: all models have drop connect rate = 0.2
    blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=False, advprop=False, verbose=True):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        advprop (bool): Whether to load pretrained weights
                        trained with advprop (valid when weights_path is None).
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])

    state_dict = convert_state_dict(state_dict)

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, f'Missing keys when loading pretrained weights: {ret.missing_keys}'
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == {'_fc.weight', '_fc.bias'}, f'Missing keys when loading pretrained weights: {ret.missing_keys}'

    assert not ret.unexpected_keys, f'Missing keys when loading pretrained weights: {ret.unexpected_keys}'

    if verbose:
        print(f'Loaded pretrained weights for {model_name}')


def convert_state_dict(state_dict):
    state_dict['_conv_stem.conv.weight'] = state_dict.pop('_conv_stem.weight')
    state_dict['_conv_stem.bn.weight'] = state_dict.pop('_bn0.weight')
    state_dict['_conv_stem.bn.bias'] = state_dict.pop('_bn0.bias')
    state_dict['_conv_stem.bn.running_mean'] = state_dict.pop('_bn0.running_mean')
    state_dict['_conv_stem.bn.running_var'] = state_dict.pop('_bn0.running_var')
    state_dict['_conv_stem.bn.num_batches_tracked'] = state_dict.pop('_bn0.num_batches_tracked')

    state_dict['_conv_head.conv.weight'] = state_dict.pop('_conv_head.weight')
    state_dict['_conv_head.bn.weight'] = state_dict.pop('_bn1.weight')
    state_dict['_conv_head.bn.bias'] = state_dict.pop('_bn1.bias')
    state_dict['_conv_head.bn.running_mean'] = state_dict.pop('_bn1.running_mean')
    state_dict['_conv_head.bn.running_var'] = state_dict.pop('_bn1.running_var')
    state_dict['_conv_head.bn.num_batches_tracked'] = state_dict.pop('_bn1.num_batches_tracked')

    for i in range(32):
        if i >= 2:
            state_dict[f'_blocks.{i}._expand_conv.conv.weight'] = state_dict.pop(f'_blocks.{i}._expand_conv.weight')
            state_dict[f'_blocks.{i}._expand_conv.bn.weight'] = state_dict.pop(f'_blocks.{i}._bn0.weight')
            state_dict[f'_blocks.{i}._expand_conv.bn.bias'] = state_dict.pop(f'_blocks.{i}._bn0.bias')
            state_dict[f'_blocks.{i}._expand_conv.bn.running_mean'] = state_dict.pop(f'_blocks.{i}._bn0.running_mean')
            state_dict[f'_blocks.{i}._expand_conv.bn.running_var'] = state_dict.pop(f'_blocks.{i}._bn0.running_var')
            state_dict[f'_blocks.{i}._expand_conv.bn.num_batches_tracked'] = state_dict.pop(f'_blocks.{i}._bn0.num_batches_tracked')
        state_dict[f'_blocks.{i}._depthwise_conv.conv.weight'] = state_dict.pop(f'_blocks.{i}._depthwise_conv.weight')
        state_dict[f'_blocks.{i}._depthwise_conv.bn.weight'] = state_dict.pop(f'_blocks.{i}._bn1.weight')
        state_dict[f'_blocks.{i}._depthwise_conv.bn.bias'] = state_dict.pop(f'_blocks.{i}._bn1.bias')
        state_dict[f'_blocks.{i}._depthwise_conv.bn.running_mean'] = state_dict.pop(f'_blocks.{i}._bn1.running_mean')
        state_dict[f'_blocks.{i}._depthwise_conv.bn.running_var'] = state_dict.pop(f'_blocks.{i}._bn1.running_var')
        state_dict[f'_blocks.{i}._depthwise_conv.bn.num_batches_tracked'] = state_dict.pop(f'_blocks.{i}._bn1.num_batches_tracked')
        state_dict[f'_blocks.{i}._se_reduce.conv.weight'] = state_dict.pop(f'_blocks.{i}._se_reduce.weight')
        state_dict[f'_blocks.{i}._se_reduce.conv.bias'] = state_dict.pop(f'_blocks.{i}._se_reduce.bias')
        state_dict[f'_blocks.{i}._se_expand.conv.weight'] = state_dict.pop(f'_blocks.{i}._se_expand.weight')
        state_dict[f'_blocks.{i}._se_expand.conv.bias'] = state_dict.pop(f'_blocks.{i}._se_expand.bias')
        state_dict[f'_blocks.{i}._project_conv.conv.weight'] = state_dict.pop(f'_blocks.{i}._project_conv.weight')
        state_dict[f'_blocks.{i}._project_conv.bn.weight'] = state_dict.pop(f'_blocks.{i}._bn2.weight')
        state_dict[f'_blocks.{i}._project_conv.bn.bias'] = state_dict.pop(f'_blocks.{i}._bn2.bias')
        state_dict[f'_blocks.{i}._project_conv.bn.running_mean'] = state_dict.pop(f'_blocks.{i}._bn2.running_mean')
        state_dict[f'_blocks.{i}._project_conv.bn.running_var'] = state_dict.pop(f'_blocks.{i}._bn2.running_var')
        state_dict[f'_blocks.{i}._project_conv.bn.num_batches_tracked'] = state_dict.pop(f'_blocks.{i}._bn2.num_batches_tracked')

    # print(state_dict.keys())
    # exit()
    # state_dict['_blocks.0._depthwise_conv.conv.weight'] = state_dict.pop('_blocks.0._depthwise_conv.conv.weight')
    return state_dict
