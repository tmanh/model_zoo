import torch
import torch.nn as nn
import torch.nn.functional as functional


class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        
        self.weights = next(self.parameters())
        self.weights.data = self.generate_weights(sparsity)
        self.weights.requires_grad_(False)

    def generate_weights(self, sparsity=0.5):
        matrix_proba = torch.float(self.weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        return binary_weights


class BlockLBP(nn.Module):
    def __init__(self, in_dim, n_weights, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_dim)
        self.conv_lbp = ConvLBP(in_dim, n_weights, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(n_weights, in_dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.batch_norm(x)
        x = functional.relu(self.conv_lbp(x))
        x = self.conv_1x1(x)
        return x.add_(residual)
