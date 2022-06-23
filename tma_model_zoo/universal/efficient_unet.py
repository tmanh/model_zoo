import torch
import torch.nn as nn

from .efficient import MBConvBlock
from .efficient_utils import GlobalParams, BlockDecoder
from ..basics.upsampling import Upsample


class EfficientUnet(nn.Module):
    def __init__(self, model, width_coefficient=None, depth_coefficient=None, image_size=None,
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
            gating=True,
        )

        self.encoder_codes = None
        self.upper_codes = None
        self.decoder_codes = None
        self.init_codes(model)

        self.encoders = self.init_blocks(self.encoder_codes)
        self.uppers = self.init_blocks(self.upper_codes)
        self.decoders = self.init_blocks(self.decoder_codes)
        
        self.upscale = Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        # encode => bottleneck => decode
        # encode
        early_states = [x]
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i in [0, 1, 2]:
                early_states.append(x)

        # bottleneck
        x = self.decoders[0](x)

        # decode
        for i in range(len(self.uppers)):
            # upscale
            height, width = early_states[-i-1].shape[2:]
            x = self.upscale(x)[:, :, :height, :width]

            # tail
            x = torch.cat([x, early_states[-i-1]], dim=1)
            x = self.uppers[i](x)
            x = self.decoders[i+1](x)

        return x

    def init_codes(self, model):
        if model == 'medium':
            self.encoder_codes = self.medium_encoder()
            self.upper_codes = self.medium_upper()
            self.decoder_codes = self.medium_decoder()
        else:
            self.encoder_codes = self.small_encoder()
            self.upper_codes = self.small_upper()
            self.decoder_codes = self.small_decoder()

    def init_blocks(self, codes):
        blocks = BlockDecoder.decode(codes)
        return nn.Sequential(*[MBConvBlock(b, self.global_params) for b in blocks])

    def medium_encoder(self):
        # r=repeats, k=kernel_size, s=stride, e=expand_ratio, i=input_filters, o=output_filters, se=se_ratio
        return ['r1_k3_s22_e1_i32_o24_noskip',
                'r2_k3_s22_e6_i24_o40_noskip',
                'r3_k3_s22_e6_i40_o112_noskip',
                'r3_k3_s22_e6_i112_o192_noskip',
                'r1_k3_s11_e6_i192_o320_noskip']

    def medium_upper(self):
        return ['r1_k3_s11_e3_i432_o192_noskip',
                'r1_k3_s11_e6_i232_o112_noskip',
                'r1_k3_s11_e6_i136_o80_noskip',
                'r1_k3_s11_e6_i112_o32_noskip']

    def medium_decoder(self):
        return ['r2_k3_s11_e6_i320_o320_noskip',
                'r2_k3_s11_e6_i192_o192_noskip',
                'r2_k3_s11_e6_i112_o112_noskip',
                'r2_k3_s11_e6_i80_o80_noskip',
                'r1_k3_s11_e6_i32_o32_noskip']

    def small_encoder(self):
        # r=repeats, k=kernel_size, s=stride, e=expand_ratio, i=input_filters, o=output_filters, se=se_ratio
        return ['r1_k3_s22_e2_i32_o24_noskip',
                'r1_k3_s22_e2_i24_o40_noskip',
                'r1_k3_s22_e2_i40_o80_noskip',
                'r1_k3_s22_e2_i80_o192_noskip',
                'r1_k3_s11_e2_i192_o256_noskip']

    def small_upper(self):
        return ['r1_k3_s11_e1_i336_o192_noskip',
                'r1_k3_s11_e1_i232_o80_noskip',
                'r1_k3_s11_e1_i104_o48_noskip',
                'r1_k3_s11_e1_i80_o32_noskip']

    def small_decoder(self):
        return ['r1_k3_s11_e1_i256_o256_noskip',
                'r1_k3_s11_e1_i192_o192_noskip',
                'r1_k3_s11_e1_i80_o80_noskip',
                'r1_k3_s11_e1_i48_o48_noskip',
                'r1_k3_s11_e1_i32_o32_noskip']


class EfficientGRUnet(nn.Module):
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
            gating=True,
        )

        self.encoders = None
        self.init_encoders()

        self.uppers = None
        self.init_uppers()

        self.decoders = None
        self.init_decoders()
        
        self.upscale = Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, he, hd):
        # encode => bottleneck => decode
        # encode
        current_he = [x]
        k = 0
        for i, encoder in enumerate(self.encoders):
            if i in [1, 2, 3, 4]:
                x = torch.cat([x, he[k]], dim=1)
                k += 1
            x = encoder(x)
            if i in [0, 1, 2, 3]:
                current_he.append(x)

        # bottleneck
        x = self.decoders[0](x)

        # decode
        current_hd = []
        for i in range(len(self.uppers)):
            # upscale
            height, width = current_he[-i-2].shape[2:]
            x = self.upscale(x)[:, :, :height, :width]

            # tail
            x = torch.cat([x, current_he[-i-2]], dim=1)
            x = self.uppers[i](x)
 
            current_hd.append(x)

            x = torch.cat([x, hd[i]], dim=1)
            x = self.decoders[i+1](x)

        return x, current_he[1:], current_hd

    def init_blocks(self, codes):
        blocks = BlockDecoder.decode(codes)
        return nn.Sequential(*[MBConvBlock(b, self.global_params) for b in blocks])

    def init_encoders(self):
        # r=repeats, k=kernel_size, s=stride, e=expand_ratio, i=input_filters, o=output_filters, se=se_ratio
        encoder_blocks_args = [
            'r2_k3_s22_e2_i32_o24_noskip',
            'r2_k3_s22_e2_i48_o40_noskip',
            'r2_k3_s22_e1_i80_o80_noskip',
            'r2_k3_s22_e1_i160_o192_noskip',
            'r1_k3_s11_e1_i384_o256_noskip']
        self.encoders = self.init_blocks(encoder_blocks_args)

    def init_uppers(self):
        # r=repeats, k=kernel_size, s=stride, e=expand_ratio, i=input_filters, o=output_filters, se=se_ratio
        upper_blocks_args = [
            'r1_k3_s11_e1_i336_o192_noskip',
            'r1_k3_s11_e1_i232_o112_noskip',
            'r1_k3_s11_e1_i136_o80_noskip',
            'r1_k3_s11_e1_i112_o32_noskip']
        self.uppers = self.init_blocks(upper_blocks_args)

    def init_decoders(self):
        # r=repeats, k=kernel_size, s=stride, e=expand_ratio, i=input_filters, o=output_filters, se=se_ratio
        decoder_blocks_args = [
            'r1_k3_s11_e1_i256_o256_noskip',
            'r1_k3_s11_e1_i384_o192_noskip',
            'r1_k3_s11_e1_i224_o112_noskip',
            'r1_k3_s11_e1_i160_o80_noskip',
            'r1_k3_s11_e1_i64_o32_noskip']
        self.decoders = self.init_blocks(decoder_blocks_args)


def test():
    x = torch.zeros(1, 3, 484, 648).cuda()
    
    """
    conv = nn.Conv2d(3, 32, 3, 1, 1, bias=False).cuda()
    x = conv(x)
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    en = EfficientUnet(model='small').cuda()
    x = en(x)
    print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    # """

    # """
    he = [torch.zeros(1, 24, 242, 324).cuda(), torch.zeros(1, 40, 121, 162).cuda(), torch.zeros(1, 80, 61, 81).cuda(), torch.zeros(1, 192, 31, 41).cuda()]
    hd = [torch.zeros(1, 192, 61, 81).cuda(), torch.zeros(1, 112, 121, 162).cuda(), torch.zeros(1, 80, 242, 324).cuda(), torch.zeros(1, 32, 484, 648).cuda()]

    conv = nn.Conv2d(3, 32, 3, 1, 1, bias=False).cuda()
    x = conv(x)
    en = EfficientGRUnet().cuda()
    
    for _ in range(4):
        x, he, hd = en(x, he, hd)
        print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    # """
