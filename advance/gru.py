import torch
import torch.nn as nn
import torch.nn.functional as functional

from advance.basics.conv_gru import ConvGRU2d, LightGRU2d, DeformableLightGRU2d


class BaseGRUUNet(nn.Module):
    def __init__(self, in_channels, enc_channels=[64, 128, 256, 512], dec_channels=[256, 128, 64], n_enc_convs=3, n_dec_convs=3, gru_all=False,
                 act=nn.ReLU(inplace=True), enc_gru_conv=ConvGRU2d, dec_gru_conv=ConvGRU2d, last_conv=ConvGRU2d, conv=nn.Conv2d):
        super().__init__()
        self.n_rnn = 0
        self.act = act
        self.enc_gru_conv = enc_gru_conv
        self.dec_gru_conv = dec_gru_conv
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.n_enc_convs = n_enc_convs
        self.n_dec_convs = n_dec_convs
        self.conv = conv
        self.last_conv = last_conv
        self.gru_all = gru_all

        self.encoders = []
        self.init_encoders()

        self.decoders = []
        self.init_decoders()

    def init_encoders(self):
        cin = self.in_channels
        stride = 1
        for cout in self.enc_channels:
            self.encoders.append(self._enc(cin, cout, stride=stride, n_convs=self.n_enc_convs, gru_all=self.gru_all))
            stride = 2
            cin = cout
        self.encoders = nn.ModuleList(self.encoders)

    def init_decoders(self):
        pass

    def _enc(self, channels_in, channels_out, stride=2, n_convs=2, gru_all=False):
        mods = []
        if stride > 1:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(self.enc_gru_conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
            else:
                mods.append(self.conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
                mods.append(self.act)
            channels_in = channels_out
            stride = 1
        return nn.Sequential(*mods)

    def _dec(self, channels_in, channels_out, n_convs=2, gru_all=False, last=False):
        mods = []
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                if last:
                    mods.append(self.last_conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
                else:
                    mods.append(self.dec_gru_conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
            else:
                mods.append(self.conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
                mods.append(self.act)
            channels_in = channels_out
        return nn.Sequential(*mods)

    def encode(self, x, hs):
        hidx = 0
        feats = []
        for enc in self.encoders:
            for mod in enc:
                if isinstance(mod, self.enc_gru_conv):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        return x, feats, hidx

    def decode(self, x, hs, feats, hidx):
        return x, hs 

    def forward(self, x, hs=None):
        if hs is None:
            hs = [None for _ in range(self.n_rnn)]

        x, feats, hidx = self.encode(x, hs)
        x, hs = self.decode(x, hs, feats, hidx)
        
        return x, hs


class GRUUNet(BaseGRUUNet):
    def __init__(self, channels_in, enc_channels=[64, 128, 256, 512], dec_channels=[256, 128, 64], n_enc_convs=3, n_dec_convs=3, gru_all=False,
                 act=nn.ReLU(inplace=True), enc_gru_conv=ConvGRU2d, dec_gru_conv=ConvGRU2d, last_conv=ConvGRU2d, conv=nn.Conv2d):
        super().__init__(channels_in, enc_channels, dec_channels, n_enc_convs, n_dec_convs, gru_all, act, enc_gru_conv, dec_gru_conv, last_conv, conv)
        
    def init_decoders(self):
        cin = self.enc_channels[-1] + self.enc_channels[-2]
        decs = []
        for idx, cout in enumerate(self.dec_channels):
            decs.append(self._dec(cin, cout, n_convs=self.n_dec_convs, gru_all=self.gru_all))
            cin = cout + self.enc_channels[max(-idx - 3, -len(self.enc_channels))]
        self.decs = nn.ModuleList(decs)

    def decode(self, x, hs, feats, hidx):
        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                if isinstance(mod, (self.dec_gru_conv, self.last_conv)):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        x = feats.pop()

        return x, hs


class MemorySavingGRUUNet(BaseGRUUNet):
    def __init__(self, channels_in, enc_channels=[64, 128, 256, 512], dec_channels=[512, 256, 128, 64], n_enc_convs=3, n_dec_convs=3, gru_all=False,
                 act=nn.ReLU(inplace=True), enc_gru_conv=ConvGRU2d, dec_gru_conv=ConvGRU2d, last_conv=ConvGRU2d, conv=nn.Conv2d):
        super().__init__(channels_in, enc_channels, dec_channels, n_enc_convs, n_dec_convs, gru_all, act, enc_gru_conv, dec_gru_conv, last_conv, conv)

    def _enc(self, channels_in, channels_out, stride=1, n_convs=2, gru_all=False):
        mods = []
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(self.enc_gru_conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
            elif idx == 0:
                mods.append(self.conv(channels_in, channels_out, kernel_size=3, stride=2, padding=1, bias=False))
                mods.append(self.act)
            else:
                mods.append(self.conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
                mods.append(self.act)
            channels_in = channels_out
        return nn.Sequential(*mods)

    def init_encoders(self):
        cin = self.in_channels
        stride = 1
        for cout in self.enc_channels:
            self.encoders.append(self._enc(cin, cout, stride=stride, n_convs=self.n_enc_convs, gru_all=self.gru_all))
            cin = cout
        self.encoders = nn.ModuleList(self.encoders)
        
    def init_decoders(self):
        cin = self.enc_channels[-1] + self.enc_channels[-2]
        decs = []
        for idx, cout in enumerate(self.dec_channels):
            decs.append(self._dec(cin, cout, n_convs=self.n_dec_convs, gru_all=self.gru_all, last=(idx==len(self.dec_channels)-1)))
            cin = cout + self.enc_channels[max(-idx - 3, -len(self.enc_channels))]
        self.decs = nn.ModuleList(decs)

    def encode(self, x, hs):
        hidx = 0
        feats = [x]
        for enc in self.encoders:
            for mod in enc:
                if isinstance(mod, self.enc_gru_conv):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)
        return x, feats, hidx

    def decode(self, x, hs, feats, hidx):
        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                if isinstance(mod, (self.dec_gru_conv, self.last_conv)):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        x = feats.pop()

        return x, hs


class ResidualGRUUNet(BaseGRUUNet):
    def __init__(self, channels_in, enc_channels=[64, 128, 256, 512], dec_channels=[512, 256, 128, 64], n_enc_convs=3, n_dec_convs=3, gru_all=False,
                 act=nn.ReLU(inplace=True), enc_gru_conv=ConvGRU2d, dec_gru_conv=ConvGRU2d, last_conv=ConvGRU2d, conv=nn.Conv2d):
        super().__init__(channels_in, enc_channels, dec_channels, n_enc_convs, n_dec_convs, gru_all, act, enc_gru_conv, dec_gru_conv, last_conv, conv)

    def _enc(self, channels_in, channels_out, stride=1, n_convs=2, gru_all=False):
        mods = []
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(self.enc_gru_conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
            elif idx == 0:
                mods.append(self.conv(channels_in, channels_out, kernel_size=3, stride=2, padding=1, bias=False))
                mods.append(self.act)
            else:
                mods.append(self.conv(channels_in, channels_out, kernel_size=3, padding=1, bias=False))
                mods.append(self.act)
            channels_in = channels_out
        return nn.Sequential(*mods)

    def init_encoders(self):
        cin = self.in_channels
        stride = 1
        for cout in self.enc_channels:
            self.encoders.append(self._enc(cin, cout, stride=stride, n_convs=self.n_enc_convs, gru_all=self.gru_all))
            cin = cout
        self.encoders = nn.ModuleList(self.encoders)
        
    def init_decoders(self):
        cin = self.enc_channels[-1] + self.enc_channels[-2]
        decs = []
        for idx, cout in enumerate(self.dec_channels):
            decs.append(self._dec(cin, cout, n_convs=self.n_dec_convs, gru_all=self.gru_all, last=(idx==len(self.dec_channels)-1)))
            cin = cout + self.enc_channels[max(-idx - 3, -len(self.enc_channels))]
        self.decs = nn.ModuleList(decs)

    def encode(self, x, hs):
        hidx = 0
        feats = [x]
        for enc in self.encoders:
            for i, mod in enumerate(enc):
                if isinstance(mod, self.enc_gru_conv):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
                if i == 0:
                    bx = x
                x = x + bx
            feats.append(x)
        return x, feats, hidx

    def decode(self, x, hs, feats, hidx):
        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                if isinstance(mod, (self.dec_gru_conv, self.last_conv)):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        x = feats.pop()

        return x, hs


def test():
    torch.set_grad_enabled(False)
    x = torch.zeros(4, 3, 484, 648).cuda()

    from advance.unet import MemorySavingUNet
    conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False).cuda()
    u2net = MemorySavingUNet(64, enc_channels=[64, 128, 256], dec_channels=[256, 128, 64], n_enc_convs=2, n_dec_convs=2).cuda()

    hs = [torch.zeros(1, 64, 242, 324).cuda(), torch.zeros(1, 128, 121, 162).cuda(), torch.zeros(1, 256, 61, 81).cuda(),
          torch.zeros(1, 512, 31, 41).cuda(), torch.zeros(1, 1024, 16, 21).cuda(),
          torch.zeros(1, 1024, 31, 41).cuda(), torch.zeros(1, 512, 61, 81).cuda(),
          torch.zeros(1, 256, 121, 162).cuda(), torch.zeros(1, 128, 242, 324).cuda(), torch.zeros(1, 64, 484, 648).cuda()]

    x = conv(x)
    x = u2net(x)
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)

    en = ResidualGRUUNet(64, enc_channels=[64, 128, 256, 512, 1024], dec_channels=[1024, 512, 256, 128, 64],
                         enc_gru_conv=LightGRU2d, dec_gru_conv=DeformableLightGRU2d, last_conv=DeformableLightGRU2d, n_enc_convs=2, n_dec_convs=2).cuda()
    for i in range(4):
        print('>>>>')
        _, hs = en(x[i:i+1, :], hs)
        print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    torch.set_grad_enabled(True)
