import torch
import torch.nn as nn

from .efficient import MBConvBlock
from .efficient_utils import GlobalParams, BlockDecoder


class EfficientResnet(nn.Module):
    def __init__(self, width_coefficient=None, depth_coefficient=None, image_size=None,
                 dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True):
        super().__init__()

        self.global_params = GlobalParams(
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            image_size=image_size,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            drop_connect_rate=drop_connect_rate,
            depth_divisor=8,
            min_depth=None,
            include_top=include_top,
            need_norm=False,
            gating=False,
        )

        blocks = BlockDecoder.decode(['r6_k3_s11_e3_i32_o32'])
        self.encoders = nn.Sequential(*[MBConvBlock(b, self.global_params) for b in blocks])

    def forward(self, x):
        return self.encoders(x)


def test():
    x = torch.zeros(1, 3, 484, 648).cuda()
    
    conv = nn.Conv2d(3, 32, 3, 1, 1, bias=False).cuda()
    x = conv(x)
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    en = EfficientResnet().cuda()
    x = en(x)
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
