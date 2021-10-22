import torch.nn as nn


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
        print(str(module), module.training)
