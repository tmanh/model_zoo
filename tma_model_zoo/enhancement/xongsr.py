# https://arxiv.org/pdf/1808.08688.pdf
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional

from tma_model_zoo.basics.dynamic_conv import SamePaddingConv2dBlock


class BaseNet(nn.Module):
    def __init__(self, device, n_feats, n_resblock, act):
        super().__init__()

        self.device = device
        self.n_feats = n_feats
        self.n_resblock = n_resblock
        self.act = act

    def forward(self, *inputs):
        pass


class VDSR(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = SamePaddingConv2dBlock(in_channels, 64, 3)
        self.conv2 = nn.Sequential(*[SamePaddingConv2dBlock(64, 64, 3) for _ in range(18)])
        self.conv3 = SamePaddingConv2dBlock(64, 1, 3, act=None)

    def forward(self, x):
        identity = x
        out = self.conv3(self.conv2(self.conv1(x)))
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

        dcnn_x4 = VDSR(1)
        dcnn_x2 = VDSR(1)

        self.dcnn = nn.ModuleList([dcnn_x2, dcnn_x4])

    def forward(self, depth_lr, color_hr):
        ttt = time.time()
        _, _, out_height, out_width = color_hr.size()

        list_coarse = []
        prev = depth_lr
        if color_hr.shape[-1] / prev.shape[-1] > 1.0:
            for i in range(len(self.dcnn)):
                if i==0 or curr.shape[-1] < color_hr.shape[-1]:
                    curr = self.sr(prev)
                    if i > 0:
                        up = functional.interpolate(prev, size=curr.shape[-2:], mode='bicubic', align_corners=True)
                        curr = (curr + up) / 2

                    prev = self.dcnn[i](curr)
                    list_coarse.append(prev)

        output = prev
        if prev.shape[-2] != out_height and prev.shape[-1] != out_width:
            output = torch.nn.functional.interpolate(prev, size=(out_height, out_width), mode='bicubic', align_corners=True)

        elapsed = time.time() - ttt

        return output, list_coarse, elapsed
