import torch.nn as nn
import torch
torch.autograd.set_detect_anomaly(True)
from ..basics.conv_gru import ConvGRU2d
from ..basics.dynamic_conv import DynamicConv2d
from .resnet import ResBlock


class ResidualGRU(nn.Module):
    def __init__(self, in_channels, n_feats, kernel_size=3, n_resblock=4, gru_all=True, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.act = act
        self.in_channels = in_channels
        self.n_resblock = n_resblock
        self.gru_all = gru_all

        self.head = DynamicConv2d(in_channels, n_feats, kernel_size, act=act)
        self.body = nn.ModuleList([ResBlock(DynamicConv2d, n_feats, kernel_size, act=act) for _ in range(n_resblock)])

        if gru_all:
            self.grus = nn.ModuleList([ConvGRU2d(n_feats, n_feats, kernel_size=kernel_size) for _ in range(n_resblock)])
            self.n_grus = n_resblock
        else:
            self.grus = ConvGRU2d(n_feats, n_feats, kernel_size=kernel_size)
            self.n_grus = 1

    def forward(self, x, hs=None):
        if hs is None:
            hs = [None for _ in range(self.n_grus)]

        x = self.head(x)

        if self.gru_all:
            new_hs = []
            for i in range(self.n_resblock):
                x = self.body[i](x)
                x = self.grus[i](x, hs[i])
                new_hs.append(x)
        else:
            for i in range(self.n_resblock):
                x = self.body[i](x)

            x = x + self.grus(x, hs[0])
            new_hs = [x]
        
        return x, new_hs
