import itertools
import torch
import torch.nn as nn
import torch.nn.functional as functional


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, output, target):
        mask = (output > 0) & (target > 0)

        _output = output[mask]
        _target = target[mask]

        if _target.shape[0] == 0:
            return 0

        g = torch.log(_output) - torch.log(_target)
        dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)

        return 0 if torch.isnan(dg) else torch.sqrt(dg)


class SIL1Loss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        mask = (output > 0) & (target > 0)
        _output = output[mask]
        _target = target[mask]
        return 0 if _target.shape[0] == 0 else torch.mean(torch.abs(_output - _target))


class DSRLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.si = SIL1Loss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, tensors):
        refine = tensors['refine'] * 255.0
        gt_depth_hr = tensors['gt_depth_hr'] * 255.0
        loss_refine = self.l1(refine, gt_depth_hr)

        loss_coarse = 0
        coarses = tensors['coarse']
        for i in range(len(coarses)):
            if coarses[i].shape[-2] != gt_depth_hr.shape[-2] or coarses[i].shape[-1] != gt_depth_hr.shape[-1]:
                tmp = functional.interpolate(gt_depth_hr, size=coarses[i].shape[-2:], mode='bicubic', align_corners=True)
                loss_coarse += self.l1(coarses[i] * 255, tmp)
            else:
                loss_coarse += self.l1(coarses[i] * 255, gt_depth_hr) 

        return loss_refine + loss_coarse


class MatterportLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.si = SILogLoss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, tensors):
        refine = tensors['refine']
        gt_depth_hr = tensors['gt_depth_hr']

        coarse = tensors['coarse'] if 'coarse' in tensors.keys() else None

        loss_refine = 0  # self.si(refine[-1], gt_depth_hr)

        for i in range(len(refine)):
            if refine[i].shape[-2] != gt_depth_hr.shape[-2] or refine[i].shape[-1] != gt_depth_hr.shape[-1]:
                tmp = functional.interpolate(gt_depth_hr, size=refine[i].shape[-2:], mode='bicubic', align_corners=True)
                loss_refine += self.si(refine[i], tmp)
            else:
                loss_refine += self.si(refine[i], gt_depth_hr)

        if not isinstance(loss_refine, float) and not isinstance(loss_refine, int)  and torch.isnan(loss_refine):
            loss_refine = 0.0

        return loss_refine


class ProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.si = SILogLoss()
        self.l1 = nn.L1Loss()

    def legacy_forward(self, tensors):
        backprj_colors = tensors['backprj_colors']
        new_depths = tensors['new_depths']

        dst_colors = tensors['dst_color']
        dst_depths = tensors['dst_depth']

        n_samples, n_views = backprj_colors.shape[:2]

        lview = 0
        lgeom = 0
        lsmooth = 0
        for i in range(n_views):
            lgeom += 0.5 * self.si((new_depths[:, i] > 0).float() * dst_depths, (dst_depths > 0).float() * new_depths[:, i])
            lview += self.compute_photometric_loss(backprj_colors[:, i], dst_colors)
            lsmooth += self.compute_smooth_loss(backprj_colors[:, i], dst_colors)

        return (lview + lsmooth + lgeom) / (n_samples * n_views)

    def forward(self, tensors):
        c_sampling_maps = tensors['c_samplings']
        
        coarse = tensors['coarse']
        refined = tensors['refined']

        src_masks = tensors['valid_mask']
        src_colors = tensors['src_color']
        src_depths = tensors['src_depth']
        dst_colors = tensors['dst_color']

        n_samples, n_views = dst_colors.shape[:2]

        lview = 0
        lgeom = self.si(src_masks * coarse, (coarse > 0).float() * src_depths)
        for i, j in itertools.product(range(n_samples), range(n_views)):
            tgt_view = dst_colors[i:i+1, j]
            src_view = src_colors[i:i+1, 0]

            c_map = c_sampling_maps[i:i+1, j]

            c_prj_view = functional.grid_sample(tgt_view, c_map, mode='bilinear', padding_mode='zeros', align_corners=True)

            lview += self.compute_photometric_loss(c_prj_view, src_view)

        return (lview + lgeom) / n_views

    def forward_l(self, tensors):
        coarse = tensors['coarse']

        src_colors = tensors['src_color']
        src_depths = tensors['src_depth']

        lgeom = 0
        lsmooth = 0

        for i in range(len(coarse)):
            d = functional.interpolate(src_depths[:, 0], size=coarse[i].shape[-2:], mode='bilinear', align_corners=True)
            c = functional.interpolate(src_colors[:, 0], size=coarse[i].shape[-2:], mode='bilinear', align_corners=True)

            lgeom = 0.5 * self.si((coarse[i] > 0).float() * d, (d > 0).float() * coarse[i])
            lsmooth = self.compute_smooth_loss(coarse[i], c)

        return lsmooth + lgeom

    def robust_l1(self, output, target, eps = 1e-3):
        return torch.sqrt(torch.pow((target - output), 2) + eps ** 2).mean()
    
    def compute_photometric_loss(self, output, target):
        l1_loss = self.robust_l1(output, target)
        ssim_loss = self.structural_consistency_loss(output, target)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def compute_smooth_loss(self, predict, image):
        predict_dy, predict_dx = self.gradient_yx(predict)
        image_dy, image_dx = self.gradient_yx(image)

        # Create edge awareness weights
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

        return 5e-3 * (smoothness_x + smoothness_y)

    def depth_consistency_loss(self, src, tgt, mask):
        delta = torch.abs(tgt - src)
        loss = torch.sum(mask * delta, dim=[1, 2, 3])
        return torch.mean(loss / torch.sum(mask))

    def gradient_yx(self, image):
        dx = image[:, :, :, :-1] - image[:, :, :, 1:]
        dy = image[:, :, :-1, :] - image[:, :, 1:, :]
        return dy, dx

    def gradient(self, depth):
        depth_dy = depth[:, :, :, 1:, :] - depth[:, :, :, :-1, :]
        depth_dx = depth[:, :, :, :, 1:] - depth[:, :, :, :, :-1]
        return depth_dx, depth_dy

    def structural_consistency_loss(self, src, tgt):
        scores = self.ssim(src, tgt)
        return scores.mean()

    def ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = torch.nn.AvgPool2d(3, 1)(x)
        mu_y = torch.nn.AvgPool2d(3, 1)(y)
        mu_xy = mu_x * mu_y
        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2

        sigma_x = torch.nn.AvgPool2d(3, 1)(x ** 2) - mu_xx
        sigma_y = torch.nn.AvgPool2d(3, 1)(y ** 2) - mu_yy
        sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

        numer = (2 * mu_xy + C1)*(2 * sigma_xy + C2)
        denom = (mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2)
        score = numer / denom

        return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)

    @staticmethod
    def create_loc_matrix(depth_value):
        device = depth_value.device
        height, width = depth_value.shape[-2:]

        y = torch.arange(start=0, end=height, device=device).view(1, height, 1).repeat(1, 1, width)       # columns
        x = torch.arange(start=0, end=width, device=device).view(1, 1, width).repeat(1, height, 1)        # rows
        o = torch.ones((1, height, width), device=device)
        z = depth_value.view(1, height, width)

        return torch.cat([x * z, y * z, z, o], dim=0).view((4, -1))

    def compute_sampling_map(self, src_depth, src_intrinsic, src_extrinsic, dst_intrinsic, dst_extrinsic):
        height, width = src_depth.shape[-2:]

        # compute location matrix
        pos_matrix = self.create_loc_matrix(src_depth).reshape(4, -1)
        pos_matrix = torch.linalg.inv((src_intrinsic @ src_extrinsic)) @ pos_matrix
        pos_matrix = dst_intrinsic @ dst_extrinsic @ pos_matrix
        pos_matrix = pos_matrix.reshape((4, height, width))

        sampling_map = (pos_matrix[:2, :, :] / (pos_matrix[2:3, :, :] + 1e-7)).reshape((1, 2, height, width))
        sampling_map = sampling_map.permute((0, 2, 3, 1))

        sampling_map[:, :, :, 0:1] = sampling_map[:, :, :, 0:1] / (width / 2) - 1
        sampling_map[:, :, :, 1:2] = sampling_map[:, :, :, 1:2] / (height / 2) - 1

        return sampling_map


class SynthesisLoss(nn.Module):
    def __init__(self, mode):
        super().__init__()

        self.mode = mode
        self.eps = 1e-7

    def forward(self, tensors):
        refine = tensors['coarse']
        dst_color = tensors['dst_color']
        coarse_views = tensors['coarse_views']

        valid_loss, tv_loss = self.compute_single_view_loss(refine, dst_color)

        for i in range(coarse_views.shape[1]):
            valid_loss += torch.abs(coarse_views[:, i] - dst_color).mean()

        return valid_loss * 1.0 + tv_loss * 0.01

    def compute_single_view_loss(self, output, target):
        valid_loss = torch.abs(output - target).mean()
        tv_loss = self.total_variation_loss(output)

        return valid_loss, tv_loss

    def compute_multiple_view_loss(self, output, target):
        valid_loss = 0
        tv_loss = 0

        n_views = output.shape[1]
        for i in range(n_views):
            _valid_loss, _tv_loss = self.compute_single_view_loss(output[:, i], target)
            valid_loss += _valid_loss
            tv_loss += _tv_loss
        valid_loss /= n_views
        tv_loss /= n_views

        return valid_loss, tv_loss

    @staticmethod
    def total_variation_loss(x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
