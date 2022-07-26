import torch

# from tma_model_zoo.efficient_unet import test
# from tma_model_zoo.efficient_resnet import test
# from tma_model_zoo.u2net import test
# from tma_model_zoo.resnet import test
# from tma_model_zoo.gru import test
# from tma_model_zoo.unet import test
# from tma_model_zoo.depth_volumes import test
# from tma_model_zoo.visibility import test
# from tma_model_zoo.universal.swin import SwinTransformerA
# from tma_model_zoo.monocular_depth.depthformer import DepthFormer
# from tma_model_zoo.enhancement.base_guided import BaseFusion, DepthEncoder
# from tma_model_zoo.universal.efficient import EfficientNet
from tma_model_zoo.enhancement.cspn_fusion import generate_average_kernel, BaseCSPNFusion

from tma_model_zoo.universal.light_swin import SwinTransformerL

model = SwinTransformerL()

color =  torch.zeros((1, 3, 256, 256))
feats, _ = model(color)
for f in feats:
    print(f.shape)
# test()

# x(torch.zeros((1, 3, 480, 640)).cuda())
