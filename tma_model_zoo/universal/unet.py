import torch
import torch.nn as nn
import torch.nn.functional as functional


class BaseUNet(nn.Module):
    def __init__(self, in_channels, enc_channels = None, dec_channels = None, n_enc_convs=3, n_dec_convs=3, act=nn.ReLU(inplace=True), conv=nn.Conv2d):
        if enc_channels is None:
            enc_channels = [64, 128, 256, 512]
        if dec_channels is None:
            dec_channels = [256, 128, 64]
        super().__init__()
        self.n_rnn = 0
        self.act = act
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.n_enc_convs = n_enc_convs
        self.n_dec_convs = n_dec_convs
        self.conv = conv

        self.encoders = []
        self.init_encoders()

        self.decoders = []
        self.init_decoders()

    def init_encoders(self):
        cin = self.in_channels
        stride = 1
        for cout in self.enc_channels:
            self.encoders.append(self._enc(cin, cout, stride=stride, n_convs=self.n_enc_convs))
            stride = 2
            cin = cout
        self.encoders = nn.ModuleList(self.encoders)

    def init_decoders(self):
        pass

    def _enc(self, in_channels, channels_out, stride=2, n_convs=2):
        mods = []
        if stride > 1:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.conv(in_channels, channels_out, kernel_size=3, padding=1, bias=False))
            mods.append(self.act)
            in_channels = channels_out
            stride = 1
        return nn.Sequential(*mods)

    def _dec(self, in_channels, channels_out, n_convs=2):
        mods = []
        for _ in range(n_convs):
            mods.extend((self.conv(in_channels, channels_out, kernel_size=3, padding=1, bias=False), self.act))

            in_channels = channels_out
        return nn.Sequential(*mods)

    def encode(self, x):
        feats = []
        for enc in self.encoders:
            for mod in enc:
                x = mod(x)
            feats.append(x)

        return x, feats

    def decode(self, x, feats):
        return x 

    def forward(self, x):
        x, feats = self.encode(x)
        x = self.decode(x, feats)
        
        return x


class UNet(BaseUNet):
    def __init__(self, in_channels, enc_channels = None, dec_channels = None, n_enc_convs=3, n_dec_convs=3, gru_all=False, act=nn.ReLU(inplace=True), conv=nn.Conv2d):
        if enc_channels is None:
            enc_channels = [64, 128, 256, 512]
        if dec_channels is None:
            dec_channels = [256, 128, 64]
        super().__init__(in_channels, enc_channels, dec_channels, n_enc_convs, n_dec_convs, act, conv)
        
    def init_decoders(self):
        cin = self.enc_channels[-1] + self.enc_channels[-2]
        decs = []
        for idx, cout in enumerate(self.dec_channels):
            decs.append(self._dec(cin, cout, n_convs=self.n_dec_convs))
            cin = cout + self.enc_channels[max(-idx - 3, -len(self.enc_channels))]
        self.decs = nn.ModuleList(decs)

    def decode(self, x, feats):
        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                x = mod(x)
            feats.append(x)

        x = feats.pop()

        return x


class MemorySavingUNet(BaseUNet):
    def __init__(self, in_channels, enc_channels = None, dec_channels = None, n_enc_convs=3, n_dec_convs=3, act=nn.ReLU(inplace=True), conv=nn.Conv2d):
        if enc_channels is None:
            enc_channels = [64, 128, 256, 512]
        if dec_channels is None:
            dec_channels = [512, 256, 128, 64]
        super().__init__(in_channels, enc_channels, dec_channels, n_enc_convs, n_dec_convs, act, conv)

    def _enc(self, in_channels, channels_out, stride=1, n_convs=2):
        mods = []
        for idx in range(n_convs):
            if idx == 0:
                mods.append(self.conv(in_channels, channels_out, kernel_size=3, stride=2, padding=1, bias=False))
            else:
                mods.append(self.conv(in_channels, channels_out, kernel_size=3, padding=1, bias=False))
            mods.append(self.act)
            in_channels = channels_out
        return nn.Sequential(*mods)

    def init_encoders(self):
        cin = self.in_channels
        stride = 1
        for cout in self.enc_channels:
            self.encoders.append(self._enc(cin, cout, stride=stride, n_convs=self.n_enc_convs))
            cin = cout
        self.encoders = nn.ModuleList(self.encoders)
        
    def init_decoders(self):
        enc_channels = [self.in_channels, *self.enc_channels]
        cin = enc_channels[-1] + enc_channels[-2]
        decs = []
        for idx, cout in enumerate(self.dec_channels):
            decs.append(self._dec(cin, cout, n_convs=self.n_dec_convs))
            cin = cout + enc_channels[max(-idx - 3, -len(enc_channels))]
        self.decs = nn.ModuleList(decs)

    def encode(self, x):
        feats = [x]
        for enc in self.encoders:
            for mod in enc:
                x = mod(x)
            feats.append(x)
        return x, feats

    def decode(self, x, feats):
        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                x = mod(x)
            feats.append(x)

        x = feats.pop()

        return x


class ResidualUNet(BaseUNet):
    def __init__(self, in_channels, enc_channels = None, dec_channels = None, n_enc_convs=3, n_dec_convs=3, act=nn.ReLU(inplace=True), conv=nn.Conv2d):
        if enc_channels is None:
            enc_channels = [64, 128, 256, 512]
        if dec_channels is None:
            dec_channels = [256, 128, 64, 64, 64]
        super().__init__(in_channels, enc_channels, dec_channels, n_enc_convs, n_dec_convs, act, conv)

    def _enc(self, in_channels, channels_out, stride=1, n_convs=2):
        mods = []
        for idx in range(n_convs):
            if idx == 0:
                mods.append(self.conv(in_channels, channels_out, kernel_size=3, stride=2, padding=1, bias=False))
            else:
                mods.append(self.conv(in_channels, channels_out, kernel_size=3, padding=1, bias=False))
            mods.append(self.act)
            in_channels = channels_out

        return nn.Sequential(*mods)

    def init_encoders(self):
        cin = self.in_channels
        stride = 1
        for cout in self.enc_channels:
            self.encoders.append(self._enc(cin, cout, stride=stride, n_convs=self.n_enc_convs))
            cin = cout

        self.encoders.append(nn.Sequential(*[self.conv(self.enc_channels[-1], self.enc_channels[-1], kernel_size=1, padding=1, bias=False)]))
        self.encoders = nn.ModuleList(self.encoders)
        
    def init_decoders(self):
        cin = self.enc_channels[-1]
        decs = []
        for cout in self.dec_channels:
            decs.append(self._dec(cin, cout, n_convs=self.n_dec_convs))
            cin = cout
        self.decs = nn.ModuleList(decs)

    def encode(self, x):
        feats = [x]

        for enc in self.encoders:
            for mod in enc:
                x = mod(x)

            feats.append(x)
        return x, feats

    def decode(self, x, feats):
        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')

            x = x0 if x0.shape[1] != x1.shape[1] else x0 + x1
            for mod in dec:
                x = mod(x)
            feats.append(x)

        x = feats.pop()

        return x


def test():
    # torch.set_grad_enabled(False)
    x = torch.zeros(4, 64, 484, 648).cuda()
    unet = MemorySavingUNet(64).cuda()
    x = unet(x)
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    # torch.set_grad_enabled(True)
