import torch.nn.functional as functional

softmax_helper = lambda x: functional.softmax(x, 1)
