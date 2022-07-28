# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from ..universal.depthformer_basics import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        

    @torch.no_grad()
    def accuracy(self, output, target, topk=(1, 5, )):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self,
                input,
                target):
        """Forward function."""

        loss_ce = F.cross_entropy(input.squeeze(), target)
        acc = self.accuracy(input.squeeze(), target)
        loss_cls = self.loss_weight * loss_ce
        return loss_cls, acc


@LOSSES.register_module()
class BinsChamferLoss(nn.Module):
    """BinsChamferLoss used in Adabins. Waiting for re-writing
    Args:
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 loss_weight=0.1):
        super(BinsChamferLoss, self).__init__()
        self.loss_weight = loss_weight

    def bins_chamfer_loss(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

    def forward(self,
                input,
                target,
                **kwargs):
        """Forward function."""
        
        chamfer_loss = self.bins_chamfer_loss(input, target)
        chamfer_loss = self.loss_weight * chamfer_loss
        return chamfer_loss


@LOSSES.register_module()
class SigLoss(nn.Module):
    """SigLoss.
    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implement for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up and self.warm_up_counter < self.warm_iter:
            g = torch.log(input + self.eps) - torch.log(target + self.eps)
            g = 0.15 * torch.pow(torch.mean(g), 2)
            self.warm_up_counter += 1
            return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""
        
        return self.loss_weight * self.sigloss(depth_pred, depth_gt,)
