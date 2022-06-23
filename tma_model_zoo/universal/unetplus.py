# https://github.com/MrGiovanni/UNetPlusPlus

import torch
import numpy as np
import torch.nn as nn

from ..utils.activations import softmax_helper

from ..utils.initialization import HeInitializer
from ..basics import ConvDropoutNormNonlin
from ..basics import StackedConvLayers
from ..basics import Upsample

from nnunet.network_architecture.neural_network import SegmentationNetwork



class UNetPlusPlus(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    def set_default_args_dict(self):
        if self.nonlin_kwargs is None:
            self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if self.dropout_op_kwargs is None:
            self.dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if self.norm_op_kwargs is None:
            self.norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

    def set_default_max_num_features(self):
        if self.max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D

    def setup_pooling_layer(self, num_pool):
        if self.pool_op_kernel_sizes is None:
            self.pool_op_kernel_sizes = [(2, 2)] * num_pool
        if self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        
        self.input_shape_must_be_divisible_by = np.prod(self.pool_op_kernel_sizes, 0, dtype=np.int64)

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weight_initializer=HeInitializer(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False):
        super(UNetPlusPlus, self).__init__()

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weight_initializer = weight_initializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        self.set_default_args_dict()

        if conv_op != nn.Conv2d:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.setup_pooling_layer(num_pool)

        upsample_mode = 'bilinear'
        pool_op = nn.MaxPool2d
        transpconv = nn.ConvTranspose2d

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.max_num_features = max_num_features
        self.set_default_max_num_features()

        self.conv_blocks_context = []

        self.loc0 = []
        self.loc1 = []
        self.loc2 = []
        self.loc3 = []
        self.loc4 = []
        self.td = []
        self.up0 = []
        self.up1 = []
        self.up2 = []
        self.up3 = []
        self.up4 = []

        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        output_features, input_features = self.append_context_layers(num_pool, num_conv_per_stage, feat_map_mul_on_downscale, pool_op_kernel_sizes, basic_block, pool_op, output_features, input_features)

        # now the bottleneck. Determine the first stride.
        first_stride = pool_op_kernel_sizes[-1] if self.convolutional_pooling else None

        # The output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        encoder_features = final_num_features
        self.loc0, self.up0, encoder_features = self.create_nest(0, num_pool, final_num_features, num_conv_per_stage, basic_block, transpconv)
        self.loc1, self.up1, encoder_features1 = self.create_nest(1, num_pool, encoder_features, num_conv_per_stage, basic_block, transpconv)
        self.loc2, self.up2, encoder_features2 = self.create_nest(2, num_pool, encoder_features1, num_conv_per_stage, basic_block, transpconv)
        self.loc3, self.up3, encoder_features3 = self.create_nest(3, num_pool, encoder_features2, num_conv_per_stage, basic_block, transpconv)
        self.loc4, self.up4, encoder_features4 = self.create_nest(4, num_pool, encoder_features3, num_conv_per_stage, basic_block, transpconv)

        self.seg_outputs.append(conv_op(self.loc0[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs.append(conv_op(self.loc1[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs.append(conv_op(self.loc2[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs.append(conv_op(self.loc3[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs.append(conv_op(self.loc4[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool):
            if self.upscale_logits:
                self.upscale_logits_ops.append(
                    Upsample(scale_factor=tuple(int(i) for i in cum_upsample[usl + 1]), mode=upsample_mode)
                )

            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.loc0 = nn.ModuleList(self.loc0)
        self.loc1 = nn.ModuleList(self.loc1)
        self.loc2 = nn.ModuleList(self.loc2)
        self.loc3 = nn.ModuleList(self.loc3)
        self.loc4 = nn.ModuleList(self.loc4)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.up0 = nn.ModuleList(self.up0)
        self.up1 = nn.ModuleList(self.up1)
        self.up2 = nn.ModuleList(self.up2)
        self.up3 = nn.ModuleList(self.up3)
        self.up4 = nn.ModuleList(self.up4)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weight_initializer is not None:
            self.apply(self.weight_initializer)

    def append_context_layers(self, num_pool, num_conv_per_stage, feat_map_mul_on_downscale, pool_op_kernel_sizes, basic_block, pool_op, output_features, input_features):
        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]

            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)
        return output_features,input_features

    def forward(self, x):
        seg_outputs = []
        x0_0 = self.conv_blocks_context[0](x)
        x1_0 = self.conv_blocks_context[1](x0_0)
        x0_1 = self.loc4[0](torch.cat([x0_0, self.up4[0](x1_0)], 1))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[-1](x0_1)))

        x2_0 = self.conv_blocks_context[2](x1_0)
        x1_1 = self.loc3[0](torch.cat([x1_0, self.up3[0](x2_0)], 1))
        x0_2 = self.loc3[1](torch.cat([x0_0, x0_1, self.up3[1](x1_1)], 1))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[-2](x0_2)))

        x3_0 = self.conv_blocks_context[3](x2_0)
        x2_1 = self.loc2[0](torch.cat([x2_0, self.up2[0](x3_0)], 1))
        x1_2 = self.loc2[1](torch.cat([x1_0, x1_1, self.up2[1](x2_1)], 1))
        x0_3 = self.loc2[2](torch.cat([x0_0, x0_1, x0_2, self.up2[2](x1_2)], 1))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[-3](x0_3)))

        x4_0 = self.conv_blocks_context[4](x3_0)
        x3_1 = self.loc1[0](torch.cat([x3_0, self.up1[0](x4_0)], 1))
        x2_2 = self.loc1[1](torch.cat([x2_0, x2_1, self.up1[1](x3_1)], 1))
        x1_3 = self.loc1[2](torch.cat([x1_0, x1_1, x1_2, self.up1[2](x2_2)], 1))
        x0_4 = self.loc1[3](torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1[3](x1_3)], 1))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[-4](x0_4)))

        x5_0 = self.conv_blocks_context[5](x4_0)
        x4_1 = self.loc0[0](torch.cat([x4_0, self.up0[0](x5_0)], 1))
        x3_2 = self.loc0[1](torch.cat([x3_0, x3_1, self.up0[1](x4_1)], 1))
        x2_3 = self.loc0[2](torch.cat([x2_0, x2_1, x2_2, self.up0[2](x3_2)], 1))
        x1_4 = self.loc0[3](torch.cat([x1_0, x1_1, x1_2, x1_3, self.up0[3](x2_3)], 1))
        x0_5 = self.loc0[4](torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up0[4](x1_4)], 1))
        seg_outputs.append(self.final_nonlin(self.seg_outputs[-5](x0_5)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    # now lets build the localization pathway BACK_UP
    def create_nest(self, z, num_pool, final_num_features, num_conv_per_stage, basic_block, transpconv):
        conv_blocks_localization = []
        tu = []
        i = 0
        for u in range(z, num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * (2 + u - z)
            if i == 0:
                unet_final_features = nfeatures_from_skip
                i += 1

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                tu.append(Upsample(scale_factor=self.pool_op_kernel_sizes[-(u + 1)], mode=self.upsample_mode))
            else:
                tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, self.pool_op_kernel_sizes[-(u + 1)],
                          self.pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        return conv_blocks_localization, tu, unet_final_features

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes

        return tmp
