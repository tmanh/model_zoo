import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True, padding=-1, dilation=1, stride=1):
    if padding == -1:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias,
                         dilation=dilation, stride=stride)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias,
                         dilation=dilation, stride=stride)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class Resnet(nn.Module):
    def __init__(self, in_dim, n_feats, kernel_size, n_resblock, out_dim, act=nn.ReLU(inplace=True), tail=False):
        super(Resnet, self).__init__()

        self.head = [default_conv(in_dim, n_feats, kernel_size), act]
        self.head = nn.Sequential(*self.head)

        # define body module
        self.body = [ResBlock(default_conv, n_feats, kernel_size, act=act) for _ in range(n_resblock)]

        self.body = nn.Sequential(*self.body)

        self.tail = None
        if tail:
            self.tail = default_conv(n_feats, out_dim, kernel_size)

    def forward(self, x):
        shallow = self.head(x)
        deep = self.body(shallow)

        if self.tail is not None:
            deep = self.tail(deep)

        if deep.shape[1] == shallow.shape[1]:
            deep = deep + shallow

        return deep

    def forward_without_head(self, shallow):
        deep = self.body(shallow)

        if self.tail is not None:
            deep = self.tail(deep)

        if deep.shape[1] == shallow.shape[1]:
            deep = deep + shallow

        return deep


def test():
    x = torch.zeros(1, 3, 484, 648).cuda()
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    en = Resnet(in_dim=3, n_feats=64, kernel_size=3, n_resblock=8, out_dim=3, act=nn.ReLU(inplace=True), tail=True).cuda()
    x = en(x)
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
