import torch
import torch.nn as nn

import lighthouse.feature_extractor.vision_encoders.slowfast_model.utils.weight_init_helper as init_helper
from lighthouse.feature_extractor.vision_encoders.slowfast_model.models import head_helper, resnet_helper, stem_helper

"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}

class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = nn.BatchNorm3d(
            dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class SlowFastModel(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, last_fc=True):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        """
        super(SlowFastModel, self).__init__()
        self.last_fc = last_fc
        self.enable_detection = False
        self.num_pathways = 2
        self._construct_network()
        init_helper.init_weights(
            self, fc_init_std=0.01, zero_init_final_bn=True
        )

    def _construct_network(self):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        """
        pool_size = _POOL1['slowfast']
        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[50]

        num_groups = 1
        width_per_group = 64
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            8 // 2
        )
        temp_kernel = _TEMPORAL_KERNEL_BASIS['slowfast']

        self.s1 = stem_helper.VideoModelStem(
            dim_in=[3, 3],
            dim_out=[width_per_group, width_per_group // 8],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // 8,
            2,
            7,
            4,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // 8,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // 8,
            ],
            dim_inner=[dim_inner, dim_inner // 8],
            temp_kernel_sizes=temp_kernel[1],
            stride=[[1, 1], [2, 2], [2, 2], [2, 2]][0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[[3, 3], [4, 4], [6, 6], [3, 3]][0],
            nonlocal_inds=[[[], []], [[], []], [[], []], [[], []]][0],
            nonlocal_group=[[1, 1], [1, 1], [1, 1], [1, 1]][0],
            nonlocal_pool=[[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]][0],
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[[1, 1], [1, 1], [1, 1], [1, 1]][0],
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // 8,
            2,
            7,
            4,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // 8,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // 8,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // 8],
            temp_kernel_sizes=temp_kernel[2],
            stride=[[1, 1], [2, 2], [2, 2], [2, 2]][1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[[3, 3], [4, 4], [6, 6], [3, 3]][1],
            nonlocal_inds=[[[], []], [[], []], [[], []], [[], []]][1],
            nonlocal_group=[[1, 1], [1, 1], [1, 1], [1, 1]][1],
            nonlocal_pool=[[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]][1],
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[[1, 1], [1, 1], [1, 1], [1, 1]][1],
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // 8,
            2,
            7,
            4,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // 8,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // 8,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // 8],
            temp_kernel_sizes=temp_kernel[3],
            stride=[[1, 1], [2, 2], [2, 2], [2, 2]][2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[[3, 3], [4, 4], [6, 6], [3, 3]][2],
            nonlocal_inds=[[[], []], [[], []], [[], []], [[], []]][2],
            nonlocal_group=[[1, 1], [1, 1], [1, 1], [1, 1]][2],
            nonlocal_pool=[[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]][2],
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[[1, 1], [1, 1], [1, 1], [1, 1]][2],
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // 8,
            2,
            7,
            4,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // 8,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // 8,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=[[1, 1], [2, 2], [2, 2], [2, 2]][3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[[3, 3], [4, 4], [6, 6], [3, 3]][3],
            nonlocal_inds=[[[], []], [[], []], [[], []], [[], []]][3],
            nonlocal_group=[[1, 1], [1, 1], [1, 1], [1, 1]][3],
            nonlocal_pool=[[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]][3],
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[[1, 1], [1, 1], [1, 1], [1, 1]][3],
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // 8,
            ],
            num_classes=400,
            pool_size=[
                [
                    32
                    // 4
                    // pool_size[0][0],
                    224 // 32 // pool_size[0][1],
                    224 // 32 // pool_size[0][2],
                ],
                [
                    32 // pool_size[1][0],
                    224 // 32 // pool_size[1][1],
                    224 // 32 // pool_size[1][2],
                ],
            ],
            dropout_rate=0.5,
            last_fc=self.last_fc
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x
