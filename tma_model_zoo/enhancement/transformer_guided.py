import time
import torch.nn as nn

from ..monocular_depth.depthformer import DepthFormer
from .base_guided import BaseFusion


class TransformerGuided(nn.Module):
    def __init__(self, requires_grad):
        super().__init__()

        self.mode = 'completion'  # estimate, completion
        
        self.depth_from_color = DepthFormer(requires_grad=(self.mode=='estimate' and requires_grad))
        self.fuse = BaseFusion(DepthFormer(in_channels=1, requires_grad=requires_grad), alpha_in_channels=[64, 96, 192, 384, 768])

    def estimate(self, color_lr):
        return self.depth_from_color(color_lr)

    def extract_feats(self, depth_lr, color_lr):
        cfeats = self.depth_from_color.extract_feats(color_lr)
        completed, dfeat = self.fuse(cfeats, depth_lr)
        return completed, dfeat

    def forward(self, depth_lr, depth_bicubic, color_lr, mask_lr):
        start = time.time()
        if self.mode == 'estimate':
            output = self.depth_from_color(color_lr)
        else:
            output, dfeat = self.extract_feats(depth_lr, color_lr)
            print(output.shape)
        elapsed = time.time() - start
        return output, elapsed
