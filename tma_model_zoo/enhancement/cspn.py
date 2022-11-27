import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)


def average_by_kernel(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    return functional.conv2d(kernel, weight, stride=1, padding=(kernel_size-1) // 2)


class CSPN(nn.Module):
    def __init__(self, kernel_size):
        super(CSPN, self).__init__()
        self.kernel_size = kernel_size

        gks = 5
        self.pad = list(range(gks**2))
        for i, j in itertools.product(range(gks), range(gks)):
            top = i
            bottom = gks-1-i
            left = j
            right = gks-1-j
            self.pad[i*gks + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

        gks2 = 3     #guide kernel size
        self.pad2 = list(range(gks2**2))
        for i, j in itertools.product(range(gks2), range(gks2)):
            top = i
            bottom = gks2-1-i
            left = j
            right = gks2-1-j
            self.pad2[i*gks2 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

        gks3 = 7     #guide kernel size
        self.pad3 = list(range(gks3**2))
        for i, j in itertools.product(range(gks3), range(gks3)):
            top = i
            bottom = gks3-1-i
            left = j
            right = gks3-1-j
            self.pad3[i*gks3 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

    def forward(self, guide_weight, hn, h0):
        half = int(0.5 * (self.kernel_size * self.kernel_size - 1))
        result_pad = list(range(self.kernel_size * self.kernel_size))
        for t in range(self.kernel_size*self.kernel_size):
            zero_pad = 0
            if(self.kernel_size==3):
                zero_pad = self.pad2[t]
            elif(self.kernel_size==5):
                zero_pad = self.pad[t]
            elif(self.kernel_size==7):
                zero_pad = self.pad3[t]
            result_pad[t] = zero_pad(h0) if (t == half) else zero_pad(hn)
        guide_result = torch.cat([result_pad[t] for t in range(self.kernel_size*self.kernel_size)], dim=1)
        guide_result = torch.sum((guide_weight.mul(guide_result)), dim=1)
        guide_result = guide_result[:, int((self.kernel_size-1)/2):-int((self.kernel_size-1)/2), int((self.kernel_size-1)/2):-int((self.kernel_size-1)/2)]

        return guide_result.unsqueeze(dim=1)


class CSPNGuidanceAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):
        guide = self.generate(feature)

        guide_sum = torch.sum(guide.abs(), dim=1, keepdim=True)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1, keepdim=True))

        half1, half2 = torch.chunk(guide, 2, dim=1)
        return torch.cat((half1, guide_mid, half2), dim=1)


class CSPNGuidanceAdaptive(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature):
        return self.softmax(self.generate(feature))


class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, guidance_kernel, modified_data, source_data):  # with standard CSPN, an addition input0 port is added
        bs, _, h, w = modified_data.shape

        # STEP 1: reshape
        input_im2col = functional.unfold(modified_data, self.kernel_size, self.dilation, self.padding, self.stride)   # N x (K x K) x (H x W)
        guidance_kernel = guidance_kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)                      # N x (K x K) x (H x W)
        source_data = source_data.view(bs, 1, h * w)                                                                 # N x 1 x (H x W)

        # STEP 2: reinforce the source data back to the modified data
        mid_index = int((self.kernel_size * self.kernel_size - 1)/ 2)
        input_im2col[:, mid_index:mid_index+1, :] = source_data
        
        # STEP 3: weighted average based on guidance kernel
        output = torch.einsum('ijk,ijk->ik', (input_im2col, guidance_kernel))  # torch.sum(input_im2col * guidance_kernel, dim=1)
        return output.view(bs, 1, h, w)


class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number

        return d_result, mask_result
