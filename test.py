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
from tma_model_zoo.monocular_depth.mine_depthformer import Color2DepthEncoder

import time
import numpy as np

from PIL import Image


img = Image.open('/scratch/antruong/workspace/myspace/datasets/Matterport/1LXtFkjw3qL_FILES/undistorted_color_images/0b22fa63d0f54a529c525afbf2e8bb25_i1_0.jpg')
depth = Image.open('/scratch/antruong/workspace/myspace/datasets/Matterport/1LXtFkjw3qL_FILES/undistorted_depth_images/0b22fa63d0f54a529c525afbf2e8bb25_d1_0.png')

width, height = depth.size

scale = 2
width = width // scale
height = height // scale

dim = (width, height)
img, depth = img.resize(dim), depth.resize(dim)

img = np.array(img).astype(np.float32)
depth = np.array(depth).astype(np.float32)
x = img.reshape((1, height, width, 3))

x = x / 255.0
color_tensor =  torch.from_numpy(x).permute(0, 3, 1, 2).cuda()
print(color_tensor.shape)

model = Color2DepthEncoder('CCT').cuda()
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

start = time.time()
x = model(color_tensor)
print(f'Elapsed time: {time.time() - start} - Memory: {torch.cuda.memory_allocated(0)/1024/1024/1024}')
