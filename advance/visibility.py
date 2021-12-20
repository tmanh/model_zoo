import numpy as np
import torch
import torch.nn as nn

from .basics.padding import same_padding
from .basics.geometry import tensor_warping
from .basics.activation import stable_softmax
from .basics.dynamic_conv import Deconv, DynamicConv2d
from .depth_volumes import DepthVolumeModel, BaseDepthVolumeModel


class VisibilityModel(BaseDepthVolumeModel):
    def __init__(self, depth_start, depth_end, depth_num, memory_saving=False):
        super().__init__(depth_start, depth_end, depth_num)
        self.memory_saving = memory_saving

        self.depth_volume_model = DepthVolumeModel(depth_start=0.5, depth_end=10, depth_num=48)
        self.refine_model = RefinementNetwork()

    def warp_images(self, src_images, sampling_maps, n_views):
        warped_imgs_srcs = []
        for d in range(self.depth_num):
            feature_list = []
            for view in range(n_views):
                sampling_map = sampling_maps[:, view, d, :, :, :]
                warped_view_feature = tensor_warping(src_images[:, view, ...], sampling_map)
                feature_list.append(warped_view_feature)

            src_features = torch.stack(feature_list, dim=0)  # src_features: [V, N, C, H, W]
            warped_imgs_srcs.append(src_features)
        warped_imgs_srcs = torch.stack(warped_imgs_srcs, dim=1)  # [N, D, 1, H, W]
        return warped_imgs_srcs

    def forward(self, src_images, ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics):
        n_samples, n_views, n_channels, height, width = src_images.shape
        depth_probs, src_weights = self.depth_volume_model(src_images, ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)

        # ======== src weights ============
        if src_weights.shape[-1] != src_images.shape[-1]:
            src_weights_full_size = nn.functional.interpolate(src_weights, size=(src_images.shape[-1], src_images.shape[-2]))
        else:
            src_weights_full_size = src_weights
        src_weights_softmax = stable_softmax(src_weights_full_size, dim=1)

        # ======== depth prob ============
        if depth_probs.shape[-1] != depth_probs.shape[-1]:
            depth_probs_full_size = nn.functional.interpolate(depth_probs, size=(src_images.shape[-1], src_images.shape[-2]))
        else:
            depth_probs_full_size = depth_probs
        depth_prob_volume_softmax = stable_softmax(depth_probs_full_size, dim=0)

        # =============================== warp images =========================================
        sampling_maps, src_masks = self.compute_sampling_maps(n_samples, n_views, ys_dst, xs_dst, ys_src, xs_src, height, width, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
        sampling_maps = sampling_maps.permute(0, 1, 2, 4, 5, 3)
        src_masks = src_masks.view(n_samples, n_views, -1, height, width)
        warped_imgs_srcs = self.warp_images(src_images, sampling_maps, n_views).permute(2, 0, 1, 3, 4, 5)
        # [D, N, B, H, W, 3], # [D, N, B, H, W, 1]

        # =============== handle source weights with masks (valid warp pixels) ===========
        src_weights_softmax = src_weights_softmax * src_masks  # [N, V, D, H, W, 1]
        src_weights_softmax_sum = torch.sum(src_weights_softmax, dim=1, keepdims=True)
        src_weights_softmax_sum_zero_add = (src_weights_softmax_sum == 0.0).float() * 1e-7
        src_weights_softmax_sum += src_weights_softmax_sum_zero_add
        src_weights_softmax = src_weights_softmax/(src_weights_softmax_sum)

        # =============== Compute aggregated images =====================================
        weighted_src_img = torch.sum(src_weights_softmax.view(n_samples, n_views, self.depth_num, 1, height, width) * warped_imgs_srcs, dim=1) # [D, B, H, W, 3]
        aggregated_img = torch.sum(weighted_src_img * depth_prob_volume_softmax, dim=1)
        warped_imgs_srcs = torch.sum(warped_imgs_srcs * depth_prob_volume_softmax.view(n_samples, 1, self.depth_num, 1, height, width), dim=1)

        # ======== generator =================================
        output_imgs = []
        self_confidences = []
        for src_index in range(n_views):
            out_img, confidence = self.refine_model(aggregated_img, warped_imgs_srcs[:, src_index])
            output_imgs.append(out_img)
            self_confidences.append(confidence)

        outputs_imgs = torch.stack(output_imgs, dim=1)
        img_confidences = torch.stack(self_confidences, dim=1)
        img_confidences_norm = img_confidences / (torch.sum(img_confidences, dim=1, keepdims=True) + 1e-7)
        final_output_img = torch.sum(outputs_imgs * img_confidences_norm, dim=1)

        return aggregated_img, final_output_img, warped_imgs_srcs


class RefinementNetwork(nn.Module):
    def __init__(self, n_channels=64, act=nn.ReLU(inplace=True)):
        super().__init__()

        layer_specs = [n_channels * 2, n_channels * 4, n_channels * 8, n_channels * 8, n_channels * 8]

        self.layers_x = self.generate_layers(n_channels, layer_specs)
        self.layers_y = self.generate_layers(n_channels, layer_specs)
        self.layers_c = self.generate_confidence_layers([n_channels * 2, *[x * 2 for x in layer_specs]])
        self.layers_d = self.generate_decoders(n_channels)

        self.deconv_color = Deconv(in_channels=n_channels, out_channels=3, kernel_size=4, stride=2)
        self.deconv_confidence = Deconv(in_channels=n_channels, out_channels=1, kernel_size=4, stride=2)

        self.act = act
    
    @staticmethod
    def generate_confidence_layers(layer_specs):
        return [
            nn.Sequential(
                *[
                    DynamicConv2d(out_channels, out_channels // 2),
                    DynamicConv2d(out_channels // 2, 1, act=nn.Sigmoid()),
                ]
            )
            for out_channels in layer_specs
        ]

    @staticmethod
    def generate_layers(n_channels, layer_specs):
        layers = [DynamicConv2d(3, n_channels, kernel_size=4, stride=2, batch_norm=False, act=nn.LeakyReLU(0.2, inplace=True),)]

        in_channels = n_channels
        for i, out_channels in enumerate(layer_specs):
            if i < len(layer_specs) - 1:
                layers.append(DynamicConv2d(in_channels, out_channels, kernel_size=4, stride=2, batch_norm=True, act=nn.LeakyReLU(0.2, inplace=True)))
            else:
                layers.append(DynamicConv2d(in_channels, out_channels, kernel_size=4, stride=2, batch_norm=True, act=None))
            in_channels = out_channels

        return layers

    @staticmethod
    def generate_decoders(n_channels):
        decode_layer_specs = [
            (n_channels * 8, 0.5),   # decoder_6: [batch, 4, 16, ngf * 8 * 2] => [batch, 8, 32, ngf * 8 * 2]
            (n_channels * 8, 0.5),   # decoder_5: [batch, 8, 32, ngf * 8 * 2] => [batch, 16, 64, ngf * 8 * 2]
            (n_channels * 4, 0.0),   # decoder_4: [batch, 16, 64, ngf * 8 * 2] => [batch, 32, 128, ngf * 4 * 2]
            (n_channels * 2, 0.0),   # decoder_3: [batch, 32, 128, ngf * 4 * 2] => [batch, 64, 256, ngf * 2 * 2]
            (n_channels, 0.0),       # decoder_2: [batch, 64, 256, ngf * 2 * 2] => [batch, 128, 512, ngf * 2 * 2]
        ]

        input_layer_specs = [
            n_channels * 8,   # decoder_6: [batch, 4, 16, ngf * 8 * 2] => [batch, 8, 32, ngf * 8 * 2]
            n_channels * 16,   # decoder_5: [batch, 8, 32, ngf * 8 * 2] => [batch, 16, 64, ngf * 8 * 2]
            n_channels * 16,   # decoder_4: [batch, 16, 64, ngf * 8 * 2] => [batch, 32, 128, ngf * 4 * 2]
            n_channels * 8,   # decoder_3: [batch, 32, 128, ngf * 4 * 2] => [batch, 64, 256, ngf * 2 * 2]
            n_channels * 4,       # decoder_2: [batch, 64, 256, ngf * 2 * 2] => [batch, 128, 512, ngf * 2 * 2]
        ]

        return [
            Deconv(
                input_layer_specs[i],
                out_channels,
                kernel_size=4,
                stride=2,
                batch_norm=True,
                p_dropout=dropout,
            )
            for i, (out_channels, dropout) in enumerate(decode_layer_specs)
        ]

    def forward(self, x, y):
        confidences = []
        merges = []
        for layer_x, layer_y, layer_c in zip(self.layers_x, self.layers_y, self.layers_c):
            x = layer_x(x)
            y = layer_y(y)
            c = torch.cat([x, y], dim=1)
            c = layer_c(c)
            m = c * x + (1 - c) * y

            confidences.append(c)
            merges.append(m)

        # ===========================================================
        # decoder
        # ===========================================================
        up_feats = None
        for i, layer_d in enumerate(self.layers_d):
            skip_layer = - i - 1
            if up_feats is None:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input_tensor = merges[-1]
            else:
                up_feats = same_padding(up_feats, merges[skip_layer])
                input_tensor = torch.cat([up_feats, merges[skip_layer]], dim=1)
            up_feats = layer_d(self.act(input_tensor))

        # retified
        rectified = self.act(up_feats)

        # decoder to generate image
        colors = self.deconv_color(rectified)
        colors = torch.tanh(colors)

        # decoder to generate confidence
        confidence = self.deconv_confidence(rectified)
        confidence = torch.sigmoid(confidence)

        return colors, confidence


def test():
    visibility_model = VisibilityModel(depth_start=0.5, depth_end=10, depth_num=48)

    src_images = torch.zeros((1, 4, 3, 320, 256))
    ys_dst = torch.from_numpy(np.array([10, 10, 10, 10])).view(1, 4)
    xs_dst = torch.from_numpy(np.array([10, 10, 10, 10])).view(1, 4)
    ys_src = torch.from_numpy(np.array([10, 15, 5, 8])).view(1, 4)
    xs_src = torch.from_numpy(np.array([11, 13, 8, 15])).view(1, 4)
    dst_intrinsics = torch.from_numpy(
        np.array([[[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]]]))
    dst_intrinsics = dst_intrinsics.view(1, 4, 4, 4).float()
    
    src_intrinsics = torch.from_numpy(
        np.array([[[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]],
                  [[1170.187988, 0.000000, 647.750000, 0.000000], [0.000000, 1170.187988, 483.750000, 0.000000], [0.000000,0.000000, 1.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000]]]))
    src_intrinsics = src_intrinsics.view(1, 4, 4, 4).float()
    
    dst_extrinsics = np.array([[0.369466, 0.113037, -0.922344, 3.848802], [0.928934, -0.070575, 0.363457, 2.352613], [-0.024011, -0.991081, -0.131079, 1.420527], [0.000000, 0.000000, 0.000000, 1.000000]])
    dst_extrinsics = np.linalg.inv(dst_extrinsics)
    dst_extrinsics = torch.from_numpy(dst_extrinsics).view(1, 4, 4).repeat(4, 1, 1).view(1, 4, 4, 4).float()

    src_extrinsic_1 = np.array([[0.350988, 0.049005, -0.935097, 3.766715], [0.934441, -0.082568, 0.346415, 2.407103], [-0.060234, -0.995380, -0.074773, 1.435615], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_2 = np.array([[0.327968, 0.044623, -0.943634, 3.820534], [0.943492, -0.065749, 0.324810, 2.387508], [-0.047549, -0.996838, -0.063665, 1.462365], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_3 = np.array([[0.357426, 0.161129, -0.919937, 3.862065], [0.933725, -0.082870, 0.348269, 2.340595], [-0.020119, -0.983448, -0.180070, 1.393208], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_4 = np.array([[0.267619, 0.219644, -0.938156, 3.879654], [0.963498, -0.068228, 0.258874, 2.366085], [-0.007149, -0.973192, -0.229886, 1.356197], [0.000000, 0.000000, 0.000000, 1.000000]])
    src_extrinsic_1 = np.linalg.inv(src_extrinsic_1)
    src_extrinsic_2 = np.linalg.inv(src_extrinsic_2)
    src_extrinsic_3 = np.linalg.inv(src_extrinsic_3)
    src_extrinsic_4 = np.linalg.inv(src_extrinsic_4)
    src_extrinsics = np.array([src_extrinsic_1, src_extrinsic_2, src_extrinsic_3, src_extrinsic_4])
    src_extrinsics = torch.from_numpy(src_extrinsics).view(1, 4, 4, 4).float()

    aggregated_img, final_output_img, warped_imgs_srcs = visibility_model(src_images, ys_dst, xs_dst, ys_src, xs_src, dst_intrinsics, dst_extrinsics, src_intrinsics, src_extrinsics)
    print(aggregated_img.shape, final_output_img.shape, warped_imgs_srcs.shape)
