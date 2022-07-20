from .cspn import *


class PENet_C2(nn.Module):
    def __init__(self, backbone):
        super(PENet_C2, self).__init__()

        self.backbone = backbone

        self.kernel_conf_layer = convbn(64, 3)
        self.mask_layer = convbn(64, 1)
        self.iter_guide_layer3 = CSPNGuidanceAccelerate(64, 3)
        self.iter_guide_layer5 = CSPNGuidanceAccelerate(64, 5)
        self.iter_guide_layer7 = CSPNGuidanceAccelerate(64, 7)

        self.kernel_conf_layer_s2 = convbn(128, 3)
        self.mask_layer_s2 = convbn(128, 1)
        self.iter_guide_layer3_s2 = CSPNGuidanceAccelerate(128, 3)
        self.iter_guide_layer5_s2 = CSPNGuidanceAccelerate(128, 5)
        self.iter_guide_layer7_s2 = CSPNGuidanceAccelerate(128, 7)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.nnupsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)
        self.CSPN7 = CSPNAccelerate(kernel_size=7, dilation=1, padding=3, stride=1)
        self.CSPN3_s2 = CSPNAccelerate(kernel_size=3, dilation=2, padding=2, stride=1)
        self.CSPN5_s2 = CSPNAccelerate(kernel_size=5, dilation=2, padding=4, stride=1)
        self.CSPN7_s2 = CSPNAccelerate(kernel_size=7, dilation=2, padding=6, stride=1)

        # CSPN
        ks = 3
        encoder3 = torch.zeros(ks**2, ks**2, ks, ks).cuda()
        kernel_range_list = list(range(ks - 1, -1, -1))
        ls = []
        for _ in range(ks):
            ls.extend(kernel_range_list)
        index = [list(range(ks**2 - 1, -1, -1)), list(range(ks**2)), [val for val in kernel_range_list for j in range(ks)], ls]

        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks**2, ks**2, ks, ks).cuda()
        kernel_range_list = list(range(ks - 1, -1, -1))
        ls = []
        for _ in range(ks):
            ls.extend(kernel_range_list)
        index = [list(range(ks**2 - 1, -1, -1)), list(range(ks**2)), [val for val in kernel_range_list for j in range(ks)], ls]

        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks**2, ks**2, ks, ks).cuda()
        kernel_range_list = list(range(ks - 1, -1, -1))
        ls = []
        for _ in range(ks):
            ls.extend(kernel_range_list)
        index = [list(range(ks**2 - 1, -1, -1)), list(range(ks**2)), [val for val in kernel_range_list for j in range(ks)], ls]

        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)

    def forward(self, rgb, d, position, K):
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        feature_s1, feature_s2, coarse_depth = self.backbone(rgb, d, position, K)
        depth = coarse_depth

        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
        mask_s2 = self.mask_layer_s2(feature_s2)
        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        kernel_conf3_s2 = self.nnupsample(kernel_conf_s2[:, 0:1, :, :])
        kernel_conf5_s2 = self.nnupsample(kernel_conf_s2[:, 1:2, :, :])
        kernel_conf7_s2 = self.nnupsample(kernel_conf_s2[:, 2:3, :, :])

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)

        depth_s2 = self.nnupsample(d_s2)
        mask_s2 = self.nnupsample(mask_s2)
        depth3 = depth5 = depth7 = depth

        mask = self.mask_layer(feature_s1)
        mask = torch.sigmoid(mask)
        mask = mask * valid_mask

        kernel_conf = self.kernel_conf_layer(feature_s1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        guide3 = self.iter_guide_layer3(feature_s1)
        guide5 = self.iter_guide_layer5(feature_s1)
        guide7 = self.iter_guide_layer7(feature_s1)

        guide3 = average_by_kernel(guide3, self.encoder3)
        guide5 = average_by_kernel(guide5, self.encoder5)
        guide7 = average_by_kernel(guide7, self.encoder7)

        guide3_s2 = average_by_kernel(guide3_s2, self.encoder3)
        guide5_s2 = average_by_kernel(guide5_s2, self.encoder5)
        guide7_s2 = average_by_kernel(guide7_s2, self.encoder7)

        guide3_s2 = self.nnupsample(guide3_s2)
        guide5_s2 = self.nnupsample(guide5_s2)
        guide7_s2 = self.nnupsample(guide7_s2)

        for _ in range(6):
            depth3 = self.CSPN3_s2(guide3_s2, depth3, coarse_depth)
            depth3 = mask_s2*depth_s2 + (1-mask_s2)*depth3
            depth5 = self.CSPN5_s2(guide5_s2, depth5, coarse_depth)
            depth5 = mask_s2*depth_s2 + (1-mask_s2)*depth5
            depth7 = self.CSPN7_s2(guide7_s2, depth7, coarse_depth)
            depth7 = mask_s2*depth_s2 + (1-mask_s2)*depth7

        depth_s2 = kernel_conf3_s2 * depth3 + kernel_conf5_s2 * depth5 + kernel_conf7_s2 * depth7
        refined_depth_s2 = depth_s2

        depth3 = depth5 = depth7 = refined_depth_s2

        for _ in range(6):
            depth3 = self.CSPN3(guide3, depth3, depth_s2)
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth_s2)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth_s2)
            depth7 = mask*d + (1-mask)*depth7

        return kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7
