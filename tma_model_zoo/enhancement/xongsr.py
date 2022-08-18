# https://arxiv.org/pdf/1808.08688.pdf
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional

from tma_model_zoo.basics.dynamic_conv import DynamicConv2d


class BaseNet(nn.Module):
    def __init__(self, device, n_feats, n_resblock, act):
        super().__init__()

        self.device = device
        self.n_feats = n_feats
        self.n_resblock = n_resblock
        self.act = act

    def forward(self, *inputs):
        pass


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        return out


class VDSR(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = [ConvReLU(64) for _ in range(18)]
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1), bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)
        
        return out + identity


# https://arxiv.org/pdf/1808.08688.pdf
class XongSR(BaseNet):
    def __init__(self, device, n_feats, n_resblock, act):
        super().__init__(device, n_feats, n_resblock, act)

        self.dcnn1 = VDSR(1)
        self.dcnn2 = VDSR(1)
        self.dcnn3 = VDSR(1)
        self.dcnn4 = VDSR(1)

    def forward(self, depth_lr):
        n_samples, n_feats, height, width = depth_lr.shape

        x1 = self.dcnn1(depth_lr).view((n_samples, n_feats, 1, height, width))
        x2 = self.dcnn2(depth_lr).view((n_samples, n_feats, 1, height, width))
        x3 = self.dcnn2(depth_lr).view((n_samples, n_feats, 1, height, width))
        x4 = self.dcnn2(depth_lr).view((n_samples, n_feats, 1, height, width))

        x = torch.cat([x1, x2, x3, x4], dim=2).contiguous().view((n_samples, n_feats, 2, 2, height, width))
        x = x.permute((0, 1, 4, 2, 5, 3)).contiguous().view((n_samples, n_feats, 2 * height, 2 * width))

        return x


class XongMSR(BaseNet):
    def __init__(self, device, n_feats, n_resblock, act):
        super().__init__(device, n_feats, n_resblock, act)

        self.sr = XongSR(device, n_feats, n_resblock, act)

        self.dcnn = VDSR(1)
        self.dcnn_refine = VDSR(1)

        self.fuse = nn.Sequential(*[nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, 1), nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 2, 3, padding=1), nn.Softmax(dim=1)])

    def forward(self, depth_lr, color_hr, depth_bicubic):
        ttt = time.time()

        list_coarse = []
        prev = depth_lr
        for i in range(2):
            if i == 0 or prev.shape[-1] < depth_bicubic.shape[-1]:
                prev = self.sr(prev)
                prev = self.dcnn(prev)
                list_coarse.append(prev)

        if len(list_coarse) == 2:
            output = self.dcnn_refine(list_coarse[-1])
        else:
            output = self.dcnn(list_coarse[-1])

        elapsed = time.time() - ttt

        return output, list_coarse, elapsed
