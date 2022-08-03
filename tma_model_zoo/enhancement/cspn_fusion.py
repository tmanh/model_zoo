from .cspn import *


def generate_average_kernel(kernel_size=3):
    encoder = torch.zeros(kernel_size**2, kernel_size**2, kernel_size, kernel_size).cuda()
    kernel_range_list = list(range(kernel_size - 1, -1, -1))
    
    ls = []
    for _ in range(kernel_size):
        ls.extend(kernel_range_list)

    index = [list(range(kernel_size**2 - 1, -1, -1)), list(range(kernel_size**2)), [val for val in kernel_range_list for j in range(kernel_size)], ls]
    encoder[index] = 1
    return nn.Parameter(encoder, requires_grad=False)


class BaseCSPNFusion(nn.Module):
    def __init__(self, n_feats=64, n_reps=6):
        super().__init__()

        self.n_reps = n_reps

        self.kernel_conf_layer = convbn(n_feats, 3)
        self.gating_layer = convbn(n_feats, 1)

        self.iter_guide_layer3 = CSPNGuidanceAccelerate(n_feats, 3)
        self.iter_guide_layer5 = CSPNGuidanceAccelerate(n_feats, 5)
        self.iter_guide_layer7 = CSPNGuidanceAccelerate(n_feats, 7)

        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)

        self.softmax = nn.Softmax(dim=1)

        self.average_kernel3 = generate_average_kernel(kernel_size=3)
        self.average_kernel5 = generate_average_kernel(kernel_size=5)
        self.average_kernel7 = generate_average_kernel(kernel_size=7)

    def forward(self, guidance_feats, base_depth, coarse_depth, valid=None):
        modified_depth3 = modified_depth5 = modified_depth7 = coarse_depth

        gating = torch.sigmoid(self.gating_layer(guidance_feats))
        if valid is not None:
            gating = gating * valid

        kernel_conf = self.softmax(self.kernel_conf_layer(guidance_feats))
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        guide3 = self.iter_guide_layer3(guidance_feats)
        guide5 = self.iter_guide_layer5(guidance_feats)
        guide7 = self.iter_guide_layer7(guidance_feats)

        guide3 = average_by_kernel(guide3, self.average_kernel3)
        guide5 = average_by_kernel(guide5, self.average_kernel5)
        guide7 = average_by_kernel(guide7, self.average_kernel7)

        for _ in range(self.n_reps):
            modified_depth3 = self.CSPN3(guide3, modified_depth3, coarse_depth)
            modified_depth3 = gating * base_depth + (1 - gating) * modified_depth3
            modified_depth5 = self.CSPN5(guide5, modified_depth5, coarse_depth)
            modified_depth5 = gating * base_depth + (1 - gating) * modified_depth5
            modified_depth7 = self.CSPN7(guide7, modified_depth7, coarse_depth)
            modified_depth7 = gating * base_depth + (1 - gating) * modified_depth7

        return kernel_conf3 * modified_depth3 + kernel_conf5 * modified_depth5 + kernel_conf7 * modified_depth7


class AdaptiveCSPNFusion(nn.Module):
    def __init__(self, n_feats=64, n_reps=6):
        super().__init__()

        self.n_reps = n_reps

        self.reduce = convbn(n_feats, 32)

        self.kernel_conf_layer = convbn(34, 3)
        self.gating_layer = convbn(34, 1)

        self.iter_guide_layer3 = CSPNGuidanceAdaptive(34, 3)
        self.iter_guide_layer5 = CSPNGuidanceAdaptive(34, 5)
        self.iter_guide_layer7 = CSPNGuidanceAdaptive(34, 7)

        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, guidance_feats, base_depth, coarse_depth, valid=None):
        modified_depth3 = modified_depth5 = modified_depth7 = coarse_depth

        reduced_feats = self.reduce(guidance_feats)

        for _ in range(self.n_reps):
            merged_feats = torch.cat([reduced_feats, modified_depth3, coarse_depth], dim=1)

            gating = torch.sigmoid(self.gating_layer(merged_feats))
            if valid is not None:
                gating = gating * valid

            modified_depth3 = gating * base_depth + (1 - gating) * modified_depth3
            modified_depth5 = gating * base_depth + (1 - gating) * modified_depth5
            modified_depth7 = gating * base_depth + (1 - gating) * modified_depth7

            guide3 = self.iter_guide_layer3(merged_feats)
            guide5 = self.iter_guide_layer5(merged_feats)
            guide7 = self.iter_guide_layer7(merged_feats)

            modified_depth3 = self.CSPN3(guide3, modified_depth3, coarse_depth)
            modified_depth5 = self.CSPN5(guide5, modified_depth5, coarse_depth)
            modified_depth7 = self.CSPN7(guide7, modified_depth7, coarse_depth)

        kernel_conf = self.softmax(self.kernel_conf_layer(merged_feats))
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        return kernel_conf3 * modified_depth3 + kernel_conf5 * modified_depth5 + kernel_conf7 * modified_depth7


class CSPNFusionUp(nn.Module):
    def __init__(self, n_feats, n_reps):
        super().__init__()

        self.base_fusion = BaseCSPNFusion(n_feats=n_feats, n_reps=n_reps)
        self.refine_fusion = BaseCSPNFusion(n_feats=n_feats, n_reps=n_reps)

    def forward(self, color_feat, prev_modified_depth, depth_from_color, raw_depth, raw_mask):
        upscaled_modified_depth = functional.interpolate(prev_modified_depth, size=raw_depth.shape[-2:], align_corners=True, mode='bilinear') 
        modified_depth = self.base_fusion(color_feat, depth_from_color, upscaled_modified_depth)
        return self.refine_fusion(color_feat, raw_depth, modified_depth, raw_mask)


class CSPNFusion(nn.Module):
    def __init__(self, in_channels, n_reps):
        super().__init__()

        self.fuses = []
        for i in range(len(in_channels)):
            if i == 0:
                self.fuses.append(BaseCSPNFusion(in_channels[i], n_reps[i]))
            else:
                self.fuses.append(CSPNFusionUp(in_channels[i], n_reps[i]))
        self.fuses = nn.ModuleList(self.fuses)

    def forward(self, color_feats, depth_from_color, raw_depth):
        raw_mask = (raw_depth > 0).float()
        modified_depth = None
        for i, fuse in enumerate(self.fuses):
            valid = raw_mask
            depth = raw_depth
            depth_fc = depth_from_color

            if valid.shape != color_feats[i].shape:
                valid = functional.interpolate(valid, size=color_feats[i].shape[-2:], align_corners=True, mode='bilinear')
                depth = functional.interpolate(depth, size=color_feats[i].shape[-2:], align_corners=True, mode='bilinear')
                depth_fc = functional.interpolate(depth_fc, size=color_feats[i].shape[-2:], align_corners=True, mode='bilinear')

            if i == 0:
                modified_depth = fuse(color_feats[i], base_depth=depth, coarse_depth=depth_fc, valid=valid)
            else:
                modified_depth = fuse(color_feats[i], prev_modified_depth=modified_depth, raw_depth=depth, depth_from_color=depth_fc, raw_mask=valid)

        return [modified_depth], color_feats[-1]
