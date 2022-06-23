import torch
import torch.nn as nn
import torch.nn.functional as functional

class GaussianFilter(nn.Module):
    def __init__(self, n_channels, ksize=5, sigma=None):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        if sigma is None:
            sigma = 0.3 * ((ksize-1) / 2.0 - 1) + 0.8
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(ksize)
        x_grid = x_coord.repeat(ksize).view(ksize, ksize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        center = ksize // 2
        weight = torch.exp(-torch.sum((xy_grid - center)**2., dim=-1) / (2*sigma**2))
        # Make sure sum of values in gaussian kernel equals 1.
        weight /= torch.sum(weight)
        self.gaussian_weight = weight

        # Reshape to 2d depthwise convolutional weight
        weight = weight.view(1, 1, ksize, ksize)
        weight = weight.repeat(n_channels, 1, 1, 1)

        # create gaussian filter as convolutional layer
        pad = (ksize - 1) // 2
        self.filter = nn.Conv2d(n_channels, n_channels, ksize, stride=1, padding=pad, groups=n_channels, bias=False, padding_mode='reflect')
        self.filter.weight.data = weight
        self.filter.weight.requires_grad = False
    
    def forward(self, x):
        return self.filter(x)


class BilateralFilter(nn.Module):
    def __init__(self, ksize=3, sigma_space=None, sigma_density=None):
        super(BilateralFilter, self).__init__()
        # initialization
        if sigma_space is None:
            self.sigma_space = 0.3 * ((ksize-1) * 0.5 - 1) + 0.8
        if sigma_density is None:
            self.sigma_density = self.sigma_space

        self.pad = (ksize-1) // 2
        self.ksize = ksize
        
        # get the spatial gaussian weight
        self.weight_space = GaussianFilter(ksize=self.ksize, sigma=self.sigma_space).gaussian_weight

    def forward(self, depth, color, mask):
        n_samples, n_views, n_channels, height, width = color.shape

        depth = depth.view(n_samples * n_views, 1, height, width)
        mask = mask.view(n_samples * n_views, 1, height, width)
        color = color.view(n_samples * n_views, n_channels, height, width)

        # Padding
        color_pad = functional.pad(color, pad=[self.pad, self.pad, self.pad, self.pad], mode='constant', value=0.0)
        depth_pad = functional.pad(depth, pad=[self.pad, self.pad, self.pad, self.pad], mode='constant', value=0.0)
        mask_pad = functional.pad(mask, pad=[self.pad, self.pad, self.pad, self.pad], mode='constant', value=0.0)

        # Extracts sliding local patches from a batched input tensor.
        color_patches = color_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        depth_patches = depth_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        mask_patches = mask_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)

        # Calculate the 2-dimensional gaussian kernel
        diff_density = depth_patches - depth.unsqueeze(-1).unsqueeze(-1)
        weight_density = torch.exp(-(diff_density ** 2) / (2 * self.sigma_density ** 2))

        # Normalization
        norm_weight_density = weight_density / weight_density.sum(dim=(-1, -2), keepdim=True)

        # Calculate the 2-dimensional gaussian kernel
        diff_density_c = color_patches - color.unsqueeze(-1).unsqueeze(-1)
        weight_density_c = torch.exp(-(diff_density_c ** 2) / (2 * self.sigma_density ** 2))
        weight_density_c = torch.sum(weight_density_c, dim=1, keepdim=True)
        norm_weight_density_c = weight_density_c / weight_density_c.sum(dim=(-1, -2), keepdim=True)

        # Keep same shape with weight_density
        weight_space = self.weight_space.view(1, 1, 1, 1, self.ksize, self.ksize).to(depth.device)

        # get the final kernel weight
        weight_patches = norm_weight_density * norm_weight_density_c * weight_space * mask_patches + 1e-7
        weight_patches_sum = weight_patches.sum(dim=(-1, -2))

        filtered = (weight_patches * color_patches).sum(dim=(-1, -2)) / weight_patches_sum
        filtered = filtered.view(n_samples, n_views, n_channels, height, width)

        return filtered
