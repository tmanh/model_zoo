import torch.nn as nn
import numpy as np


def print_module_training_status(module):
    if isinstance(
        module,
        (
            nn.Conv2d,
            nn.Conv3d,
            nn.Dropout3d,
            nn.Dropout2d,
            nn.Dropout,
            nn.InstanceNorm3d,
            nn.InstanceNorm2d,
            nn.InstanceNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.BatchNorm1d,
        ),
    ):
        print(module, module.training)


def tensor2images(tensors, min_value, max_value):
    if len(tensors.shape) != 4:
        return []

    n_samples = tensors.shape[0]
    return [tensor2image(tensors[i], min_value, max_value) for i in range(n_samples)]


def tensor2image(input_tensor, min_value, max_value):
    channels, height, width = input_tensor.shape
    image = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min()) * (max_value - min_value) + min_value

    if channels == 1:
        return image.view(height, width).detach().cpu().numpy().astype(np.uint8)
    return image.permute(1, 2, 0).view(height, width, channels).detach().cpu().numpy().astype(np.uint8)


def is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f'invalid input for is_power_of_2: {n} (type: {type(n)})')
    return (n & (n - 1) == 0) and n != 0


def freeze_module(a_module):
    for module in a_module.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.momentum = 0
    for param in a_module.parameters():
        param.requires_grad = False
    a_module.eval()
