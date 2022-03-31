# https://arxiv.org/pdf/1808.08688.pdf
import time
import torch
import torch.nn as nn

from tma_model_zoo.basics.dynamic_conv import SamePaddingConv2dBlock
from tma_model_zoo.universal.resnet import ResBlock


class BaseNet(nn.Module):
    def __init__(self, device, n_feats, n_resblock, act):
        super().__init__()

        self.device = device
        self.n_feats = n_feats
        self.n_resblock = n_resblock
        self.act = act

    def forward(self, *inputs):
        pass


# https://arxiv.org/pdf/1808.08688.pdf
class XongSR(BaseNet):
    def __init__(self, device, n_feats, n_resblock, act):
        super().__init__(device, n_feats, n_resblock, act)

        self.dcnn1 = [SamePaddingConv2dBlock(1, n_feats, 3)]
        self.dcnn1.extend([ResBlock(SamePaddingConv2dBlock, n_feats, 3, act=nn.ReLU(inplace=True)) for _ in range(n_resblock)])
        self.dcnn1.append(SamePaddingConv2dBlock(n_feats, 1, 3))
        self.dcnn1 = nn.Sequential(*self.dcnn1)

        self.dcnn2 = [SamePaddingConv2dBlock(1, n_feats, 3)]
        self.dcnn2.extend([ResBlock(SamePaddingConv2dBlock, n_feats, 3, act=nn.ReLU(inplace=True)) for _ in range(n_resblock)])
        self.dcnn2.append(SamePaddingConv2dBlock(n_feats, 1, 3))
        self.dcnn2 = nn.Sequential(*self.dcnn2)

        self.dcnn3 = [SamePaddingConv2dBlock(1, n_feats, 3)]
        self.dcnn3.extend([ResBlock(SamePaddingConv2dBlock, n_feats, 3, act=nn.ReLU(inplace=True)) for _ in range(n_resblock)])
        self.dcnn3.append(SamePaddingConv2dBlock(n_feats, 1, 3))
        self.dcnn3 = nn.Sequential(*self.dcnn3)

        self.dcnn4 = [SamePaddingConv2dBlock(1, n_feats, 3)]
        self.dcnn4.extend([ResBlock(SamePaddingConv2dBlock, n_feats, 3, act=nn.ReLU(inplace=True)) for _ in range(n_resblock)])
        self.dcnn4.append(SamePaddingConv2dBlock(n_feats, 1, 3))
        self.dcnn4 = nn.Sequential(*self.dcnn4)

    def forward(self, depth_lr):
        n_samples, n_feats, height, width = depth_lr.size()

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

        dcnn_x4 = [SamePaddingConv2dBlock(2, n_feats, 3)]
        dcnn_x4.extend([ResBlock(SamePaddingConv2dBlock, n_feats, 3, act=nn.ReLU(inplace=True)) for _ in range(n_resblock)])
        dcnn_x4.append(SamePaddingConv2dBlock(n_feats, 1, 3))
        dcnn_x4 = nn.Sequential(*dcnn_x4)

        dcnn_x2 = [SamePaddingConv2dBlock(1, n_feats, 3)]
        dcnn_x2.extend([ResBlock(SamePaddingConv2dBlock, n_feats, 3, act=nn.ReLU(inplace=True)) for _ in range(n_resblock)])
        dcnn_x2.append(SamePaddingConv2dBlock(n_feats, 1, 3))
        dcnn_x2 = nn.Sequential(*dcnn_x2)

        self.dcnn = nn.ModuleList([dcnn_x2, dcnn_x4])

    def forward(self, depth_lr, color_hr, scale):
        ttt = time.time()
        _, _, out_height, out_width = color_hr.size()

        list_coarse = []
        prev = depth_lr
        for i in range(self.dcnn):
            curr = self.sr(prev)
            if i > 0:
                curr = torch.cat([curr, prev])

            prev = self.dcnn[i](curr)
            list_coarse.append(prev)
        
        output = torch.nn.functional.interpolate(prev, size=(out_height, out_width), mode='bicubic')

        elapsed = time.time() - ttt

        return output, list_coarse, elapsed
