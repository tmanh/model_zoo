import torch


def stable_softmax(x, dim=1):
    x = x - torch.max(x, dim=dim, keepdims=True)[0]
    x = x - torch.logsumexp(x, dim, keepdims=True)
    x = torch.exp(x)
    return x
