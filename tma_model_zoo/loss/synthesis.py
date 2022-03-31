import torch
import torch.nn as nn


class SynthesisLoss(nn.Module):
    def __init__(self, mode):
        super().__init__()

        self.mode = mode
        self.eps = 1e-7

    def forward(self, deep_images, colors, depths, poses, valid_mask):
        target = colors[:, 0, :, :, :]

        loss_tv = 0
        valid_loss = 0

        for i in range(len(deep_images)):
            dif = deep_images[i] - target
            valid_loss += torch.abs(dif).mean()
            # invalid_loss += torch.abs(dif * (1 - valid_mask)).mean()
            loss_tv += self.total_variation_loss(deep_images[i])

        return valid_loss * 1.0 + loss_tv * 0.01
        
    @staticmethod
    def total_variation_loss(x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
