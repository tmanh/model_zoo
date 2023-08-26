import logging
import warnings
import matplotlib

import torch.nn.functional as functional

from collections import OrderedDict


def swin_convert(ckpt):
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            new_k = k.replace('proj', 'projection') if 'proj' in k else k
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v
    return new_ckpt


def get_root_logger(log_file=None, log_level=logging.INFO, name='mmcv'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = logging.getLogger()

    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger



def resize(input_tensor, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=False):
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input_tensor.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if (output_h > input_h or output_w > output_h) and ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):
            warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {(input_h, input_w)} is `x+1` and out size {(output_h, output_w)} is `nx+1`')
    return functional.interpolate(input_tensor, size, scale_factor, mode, align_corners)


# color the depth, kitti magma_r, nyu jet
def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    value = (value - vmin) / (vmax - vmin) if vmin != vmax else value * 0.
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :, :3] # bgr -> rgb
    return value[..., ::-1]


def add_prefix(inputs, prefix):
    """Add prefix for dict.
    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.
    Returns:
        dict: The dict with keys updated with ``prefix``.
    """

    return {f'{prefix}.{name}': value for name, value in inputs.items()}
