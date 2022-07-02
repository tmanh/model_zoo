import torch

# from tma_model_zoo.efficient_unet import test
# from tma_model_zoo.efficient_resnet import test
# from tma_model_zoo.u2net import test
# from tma_model_zoo.resnet import test
# from tma_model_zoo.gru import test
# from tma_model_zoo.unet import test
# from tma_model_zoo.depth_volumes import test
# from tma_model_zoo.visibility import test
from tma_model_zoo.monocular_depth.depthformer import DepthFormer
from tma_model_zoo.enhancement.base_guided import BaseFusion, DepthEncoder
# from tma_model_zoo.universal.efficient import EfficientNet

# test()

df = DepthFormer(requires_grad=False).cuda()
cfeats = df.extract_feats(torch.zeros((1, 3, 480, 640)).cuda())

alpha_in_channels = [64, 96, 192, 384, 768]
de = DepthEncoder(alpha_in_channels, requires_grad=True)
det = DepthFormer(in_channels=1, requires_grad=True)
x = BaseFusion(depth_encoder=det, alpha_in_channels=alpha_in_channels).cuda()
x(cfeats, torch.zeros((1, 1, 480, 640)).cuda())
# x(torch.zeros((1, 3, 480, 640)).cuda())
