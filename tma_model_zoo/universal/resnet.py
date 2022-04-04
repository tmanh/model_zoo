import torch
import torch.nn as nn

from ..basics.dynamic_conv import DynamicConv2d


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, act=nn.ReLU(inplace=True), batch_norm=False):
        super().__init__()
        self.body = nn.Sequential(*[conv(n_feats, n_feats, kernel_size, bias=bias, act=None, batch_norm=batch_norm), act,
                                    conv(n_feats, n_feats, kernel_size, bias=bias, act=None, batch_norm=batch_norm)])

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class Resnet(nn.Module):
    def __init__(self, in_dim, n_feats, kernel_size, n_resblock, out_dim, act=nn.ReLU(inplace=True), tail=False, batch_norm=False):
        super(Resnet, self).__init__()

        self.head = DynamicConv2d(in_dim, n_feats, kernel_size, batch_norm=batch_norm)

        self.body = [ResBlock(DynamicConv2d, n_feats, kernel_size, act=act, batch_norm=batch_norm) for _ in range(n_resblock)]
        self.body = nn.Sequential(*self.body)

        self.tail = None
        if tail:
            self.tail = DynamicConv2d(n_feats, out_dim, kernel_size, batch_norm=batch_norm)

    def forward(self, x):
        shallow = self.head(x)
        return self.compute_output(shallow)

    def forward_without_head(self, shallow):
        return self.compute_output(shallow)

    def compute_output(self, shallow):
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
