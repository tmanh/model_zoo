import torch
import torch.nn as nn
import torch.nn.functional as functional

import torchvision

from math import exp
from torch.autograd import Variable


def mse_with_counting(target, output, mask=None):
    _, _, h, w = target.size()

    count = h * w
    if mask is not None:
        target_clone = torch.zeros_like(target)
        output_clone = torch.zeros_like(target)

        target_clone = target_clone.copy_(target)
        output_clone = output_clone.copy_(output[:, :, :h, :w])

        target_clone[~mask] = 0.
        output_clone[~mask] = 0.

        diff = target_clone - output_clone
        count = mask.float().sum()
    else:
        diff = target - output[:, :, :h, :w]

    return diff.pow(2).sum(), count


def mae_with_counting(target, output, mask):
    _, _, h, w = target.size()

    count = h * w
    if mask is not None:
        target_clone = torch.zeros_like(target)
        output_clone = torch.zeros_like(target)

        target_clone = target_clone.copy_(target)
        output_clone = output_clone.copy_(output[:, :, :h, :w])

        target_clone[~mask] = 0.
        output_clone[~mask] = 0.

        diff = target_clone - output_clone
        count = mask.float().sum()
    else:
        diff = target - output[:, :, :h, :w]

    return torch.abs(diff).sum(), count


##################### PSNR #####################

def psnr(img1, img2, mask, mode='int'):
    _, _, h, w = img1.size()
    
    img1 = img1[:, :, :h, :w]
    img2 = img2[:, :, :h, :w]
    
    mask = (1 - mask[:, :, :h, :w]) * (img1 != 0).float()

    if mode == 'int':
        img1 = img1.int()
        img2 = img2.int()
    else:
        img1 = img1.float()
        img2 = img2.float()

    mse = torch.sum((mask * (img1 - img2)) ** 2) / torch.sum(mask)
    
    return 10 * torch.log10(255.0 * 255.0 / mse)


##################### SSIM #####################

def ssim(target, output, mask, window_size=11, size_average=True):
    _, channel, h, w = target.shape

    window = create_window(window_size, channel)

    img1 = torch.zeros_like(target)
    img2 = torch.zeros_like(target)

    img1 = img1.copy_(target)
    img2 = img2.copy_(output[:, :, :h, :w])

    img1[~mask] = 0.
    img2[~mask] = 0.

    return _ssim(img1, img2, window, window_size, channel, size_average).detach().cpu().numpy()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    padding = window_size // 2

    mu1 = functional.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = functional.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = functional.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = functional.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = functional.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

##################### HIT MSE #####################

def hit_mse_with_counting(output, target, mask):
    _, _, h, w = target.size()

    img1 = torch.zeros_like(target)
    img2 = torch.zeros_like(target)

    img1 = img1.copy_(target)
    img2 = img2.copy_(output[:, :, :h, :w])

    img1[~mask] = 0.0
    img2[~mask] = 0.0

    output_over_render, render_over_output = img1 / img2, img2 / img1

    hit_map = torch.max(output_over_render, render_over_output) < 1.25 ** 3

    mse = (img1[hit_map] - img2[hit_map]).pow(2).sum()
    count = torch.sum(hit_map)

    return mse, count


def miss(output, target, mask, threshold):
    _, _, h, w = target.size()

    img1 = torch.zeros_like(target)
    img2 = torch.zeros_like(target)

    img1 = img1.copy_(target)
    img2 = img2.copy_(output[:, :, :h, :w])

    img1[~mask] = 0.0
    img2[~mask] = 0.0

    output_over_render, render_over_output = img1 / img2, img2 / img1

    output_over_render[torch.isnan(output_over_render)] = 0
    render_over_output[torch.isnan(render_over_output)] = 0

    miss_map = torch.max(output_over_render, render_over_output)

    return torch.sum(miss_map[mask] < threshold)


def hit_rate(target, output, mask):
    delta_105 = miss(target, output, mask, 1.05)
    delta_110 = miss(target, output, mask, 1.10)
    delta_125_1 = miss(target, output, mask, 1.25)
    delta_125_2 = miss(target, output, mask, 1.25 ** 2)
    delta_125_3 = miss(target, output, mask, 1.25 ** 3)

    return delta_105, delta_110, delta_125_1, delta_125_2, delta_125_3


##################### SSIM #####################


class VGGPerceptualLoss(nn.Module):
    def __init__(self, inp_scale="-11", mode='vgg19'):
        super().__init__()
        self.inp_scale = inp_scale
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.mode = mode

        if mode == 'vgg19':
            self.vgg = torchvision.models.vgg19(pretrained=True).features
        else:
            self.vgg = torchvision.models.vgg16(pretrained=True).features

    def forward(self, tensors):
        es = tensors['coarse']
        ta = tensors['dst_color']

        self.vgg = self.vgg.to(es.device)
        self.mean = self.mean.to(es.device)
        self.std = self.std.to(es.device)

        es = (es - self.mean) / self.std
        ta = (ta - self.mean) / self.std

        if self.mode == 'vgg19':
            return self.compute_loss_vgg19(es, ta)
        return self.compute_loss_vgg16(es, ta)

    def compute_loss_vgg16(self, es, ta):
        loss = 0
        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)

            if midx in [3, 8, 15, 22]:
                loss += torch.abs(es - ta).mean()
        return loss

    def compute_loss_vgg19(self, es, ta):
        loss = 0
        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)

            if midx in [13, 22]:
                lam = 0.5
                loss += torch.abs(es - ta).mean() * lam
            elif midx == 3:
                lam = 1
                loss += torch.abs(es - ta).mean() * lam
            elif midx == 31:
                lam = 1
                loss += torch.abs(es - ta).mean() * lam
                break
            elif midx == 8:
                lam = 0.75
                loss += torch.abs(es - ta).mean() * lam
        return loss
