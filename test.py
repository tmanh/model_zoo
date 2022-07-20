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

model = BaseCSPNFusion(n_feats=3).cuda()

guidance_feats =  torch.zeros((1, 3, 5, 5)).cuda()
guided_depth = torch.zeros((1, 1, 5, 5)).cuda()
coarse_depth = torch.zeros((1, 1, 5, 5)).cuda()
valid = torch.zeros((1, 1, 5, 5)).cuda()
x = model(guidance_feats, guided_depth, coarse_depth, valid)
print(x.shape)
# test()

# x(torch.zeros((1, 3, 480, 640)).cuda())
