import torch
import torch.nn as nn

import math

# https://github.com/xuebinqin/U-2-Net


__all__ = ['U2NET_full', 'U2NET_lite']


def upsample_like(x, size):
    return nn.Upsample(size=size, mode='bilinear', align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample

    size = list(x.shape[-2:])
    sizes = {}

    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]

    return sizes


class REBNCONV(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dilate=1, act=nn.ReLU(inplace=True)):
        super(REBNCONV, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False, act=nn.ReLU(inplace=True)):
        super(RSU, self).__init__()

        self.name = name
        self.height = height
        self.dilated = dilated

        self._make_layers(height, in_ch, mid_ch, out_ch, dilated, act=act)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height >= self.height:
                return getattr(self, f'rebnconv{height}')(x)

            x1 = getattr(self, f'rebnconv{height}')(x)

            if not self.dilated and height < self.height - 1:
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
            else:
                x2 = unet(x1, height + 1)

            x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))

            return upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x

        return x + unet(x)

    def _make_layers(self, height, in_channels, mid_channels, out_channels, dilated=False, act=nn.ReLU(inplace=True)):
        self.add_module('rebnconvin', REBNCONV(in_channels, out_channels, act=act))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.add_module('rebnconv1', REBNCONV(out_channels, mid_channels, act=act))
        self.add_module('rebnconv1d', REBNCONV(mid_channels * 2, out_channels, act=act))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)

            self.add_module(f'rebnconv{i}', REBNCONV(mid_channels, mid_channels, dilate=dilate, act=act))
            self.add_module(f'rebnconv{i}d', REBNCONV(mid_channels * 2, mid_channels, dilate=dilate, act=act))

        dilate = 2 if not dilated else 2 ** (height - 1)

        self.add_module(f'rebnconv{height}', REBNCONV(mid_channels, mid_channels, dilate=dilate, act=act))


class U2NET(nn.Module):
    def __init__(self, cfgs, out_channels, act=nn.ReLU(inplace=True)):
        super(U2NET, self).__init__()

        self.out_ch = out_channels
        self._make_layers(cfgs, act)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return upsample_like(x, sizes[height - 1])
        
        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()

            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)

            maps.insert(0, x)

            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs, act):
        self.height = int((len(cfgs) + 1) / 2)

        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1], act=act))

            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.out_ch, 3, padding=1))

        # build fuse layer
        self.add_module('outconv', nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1))


def U2NET_full():
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 128), -1],
        'stage3': ['En_3', (5, 128, 64, 256), -1],
        'stage4': ['En_4', (4, 256, 128, 512), -1],
        'stage5': ['En_5', (4, 512, 256, 512, True), -1],
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],
        'stage3d': ['De_3', (5, 512, 64, 128), 128],
        'stage2d': ['De_2', (6, 256, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }

    return U2NET(cfgs=full, out_channels=3)


def U2NET_medium():
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 128), -1],
        'stage3': ['En_3', (5, 128, 64, 256), -1],
        'stage4': ['En_4', (4, 256, 128, 256), -1],
        'stage5': ['En_5', (4, 256, 128, 256, True), -1],
        'stage6': ['En_6', (4, 256, 128, 256, True), 256],
        'stage5d': ['De_5', (4, 512, 256, 256, True), 256],
        'stage4d': ['De_4', (4, 512, 128, 128), 128],
        'stage3d': ['De_3', (5, 384, 64, 64), 64],
        'stage2d': ['De_2', (6, 192, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 32, 64), 64],
    }

    return U2NET(cfgs=lite, out_channels=3)


def U2NET_lite():
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 16, 64), -1],
        'stage2': ['En_2', (6, 64, 16, 64), -1],
        'stage3': ['En_3', (5, 64, 16, 64), -1],
        'stage4': ['En_4', (4, 64, 16, 64), -1],
        'stage5': ['En_5', (4, 64, 16, 64, True), -1],
        'stage6': ['En_6', (4, 64, 16, 64, True), 64],
        'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
        'stage4d': ['De_4', (4, 128, 16, 64), 64],
        'stage3d': ['De_3', (5, 128, 16, 64), 64],
        'stage2d': ['De_2', (6, 128, 16, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }

    return U2NET(cfgs=lite, out_channels=3)


def test():
    x = torch.zeros(1, 3, 484, 648).cuda()

    en = U2NET_medium().cuda()

    for _ in range(4):
        x = en(x)[0]
        print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
