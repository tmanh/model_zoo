import time
import torch.nn as nn

from ..monocular_depth.depthformer import DepthFormer
from .base_guided import BaseFusion, DepthEncoder
from .cspn_fusion import CSPNFusion


class TransformerGuided(nn.Module):
    def __init__(self, requires_grad):
        super().__init__()

        self.mode = 'estimate'  # estimate, completion

        # self.depth_from_color = DepthFormer(requires_grad=(False if self.mode == 'completion' else requires_grad))
        self.depth_from_color = DepthFormer(requires_grad=requires_grad)
        alpha_in_channels = self.depth_from_color.encode.list_feats
        self.fuse = BaseFusion(DepthEncoder(depth_in_channels=alpha_in_channels, requires_grad=requires_grad), alpha_in_channels=alpha_in_channels)  # [64, 96, 192, 384, 768]
        # self.fuse = BaseFusion(DepthFormer(in_channels=1, requires_grad=requires_grad), alpha_in_channels=alpha_in_channels)  # [64, 96, 192, 384, 768]
        # self.fuse = CSPNFusion(in_channels=alpha_in_channels[::-1], n_reps=[1, 2, 3, 4, 5, 6])

    def estimate(self, color_lr):
        return self.depth_from_color(color_lr)

    def extract_feats(self, depth_lr, color_lr):
        estimated, cfeats = self.depth_from_color.extract_feats(color_lr)
        completed, dfeat = self.fuse(cfeats, depth_lr)
        # completed, dfeat = self.fuse(cfeats, estimated[0], depth_lr) 
        return completed, estimated, dfeat

    def forward(self, depth_lr, depth_bicubic, color_lr, mask_lr):
        start = time.time()
        if self.mode == 'estimate':
            estimated = self.depth_from_color(color_lr)
            return None, estimated, time.time() - start
        else:
            completed, estimated, dfeat = self.extract_feats(depth_lr, color_lr)
            return completed, estimated, time.time() - start
