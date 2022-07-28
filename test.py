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

from tma_model_zoo.monocular_depth.depthformer import build_depther_from
from mmcv.runner import load_checkpoint

import cv2
import numpy as np

from PIL import Image


flag = True

if flag:
    model = build_depther_from('/scratch/antruong/workspace/myspace/model_zoo/tma_model_zoo/universal/configs/depthformer/depthformer_swint_w7_nyu.py').cuda()  # DepthFormerSwin()
    checkpoint = load_checkpoint(model, './pretrained/depthformer_swint_w7_nyu.pth', map_location='cpu')
else:
    model = build_depther_from('/scratch/antruong/workspace/myspace/model_zoo/tma_model_zoo/universal/configs/binsformer/binsformer_swint_w7_nyu.py').cuda()  # DepthFormerSwin()
    checkpoint = load_checkpoint(model, './pretrained/binsformer_swint_nyu_converted.pth', map_location='cpu')

model.eval()

img = Image.open('/scratch/antruong/workspace/myspace/datasets/Matterport/1LXtFkjw3qL_FILES/undistorted_color_images/0b22fa63d0f54a529c525afbf2e8bb25_i1_0.jpg')
depth = Image.open('/scratch/antruong/workspace/myspace/datasets/Matterport/1LXtFkjw3qL_FILES/undistorted_depth_images/0b22fa63d0f54a529c525afbf2e8bb25_d1_0.png')

width, height = depth.size

width = width // 2
height = height // 2
dim = (width, height)
img, depth = img.resize(dim), depth.resize(dim),

img = np.array(img).astype(np.float32)
depth = np.array(depth).astype(np.float32)
x = img.reshape((1, height, width, 3))

x = x / 255.0
color_tensor =  torch.from_numpy(x).permute(0, 3, 1, 2).cuda()

if flag:
    o = model.simple_run(color_tensor)
    o = o.detach().cpu().numpy().reshape((height // 2, width // 2))
else:
    o = model.simple_run(color_tensor)[0][-1]
    o = o.detach().cpu().numpy().reshape((height // 4, width // 4))

o = (o - o.min()) / (o.max() - o.min()) * 255.0
o = o.astype(np.uint8)
o = cv2.resize(o, dsize=(width, height), interpolation=cv2.INTER_AREA)

cv2.imshow('o', o)
cv2.imshow('img', img.astype(np.uint8))
cv2.waitKey()
# test()

# x(torch.zeros((1, 3, 480, 640)).cuda())
