# Copyright (c) GrokCV. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes, get_box_tensor, get_box_wh, scale_boxes
from mmdet.utils import (
    ConfigType,
    InstanceList,
    PixelList,
    MultiConfig,
    OptInstanceList,
    RangeType,
    reduce_mean,
)
from mmdet.models.dense_heads import FCOSHead
from mmdet.models.utils import multi_apply
from mmdet.models.utils import (
    filter_scores_and_topk,
    select_single_mlvl,
    unpack_gt_instances,
)

from deepir.registry import MODELS
from deepir.models.dense_heads import FCOSSegHead
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmengine.model import BaseModule, Sequential
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout

INF = 1e8


class conv_bn_relu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_bn_relu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


@MODELS.register_module()
class FCOSChangerSegHead(FCOSSegHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Strides of points
            in multiple feature levels. Defaults to (4, 8, 16, 32, 64).
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling.
            Defaults to False.
        center_sample_radius (float): Radius of center sampling.
            Defaults to 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets with
            FPN strides. Defaults to False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness
            loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self, in_channels=256, **kwargs) -> None:
        super().__init__(in_channels=in_channels, **kwargs)
        self.mlp_channels = in_channels
        self.reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.feat_channels // self.group_channels
        """Initialize classification conv layers of the head."""
        self.conv_squeeze_tgt = ConvModule(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels // 2,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )
        self.conv_squeeze_bg = ConvModule(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels // 2,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )
        self.conv_mix = ConvModule(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )
        self.conv_reduce = ConvModule(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels // self.reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )
        self.gen_weight = ConvModule(
            in_channels=self.feat_channels // self.reduction_ratio,
            out_channels=3**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
        )
        self.unfold = nn.Unfold(3, 1, (3 - 1) // 2, 1)

        self.fusion_conv = ConvModule(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

        self.convs_cls = nn.Sequential(
            conv_bn_relu(self.feat_channels, self.feat_channels // 2),
            conv_bn_relu(self.feat_channels // 2, self.feat_channels),
            # conv_bn_relu(self.feat_channels, self.feat_channels),
            # conv_bn_relu(self.feat_channels // 2, self.feat_channels)
        )
        self.convs_seg = nn.Sequential(
            conv_bn_relu(self.feat_channels, self.feat_channels // 2),
            conv_bn_relu(self.feat_channels // 2, self.feat_channels),
            # conv_bn_relu(self.feat_channels, self.feat_channels),
            # conv_bn_relu(self.feat_channels // 2, self.feat_channels)
        )
        self.convs_reg = nn.Sequential(
            conv_bn_relu(self.feat_channels, self.feat_channels // 2),
            conv_bn_relu(self.feat_channels // 2, self.feat_channels),
            # conv_bn_relu(self.feat_channels, self.feat_channels),
            # conv_bn_relu(self.feat_channels // 2, self.feat_channels)
        )

        self.ChannelEx = ChannelAttExchange(p=1 / 2, mlp_channels=self.mlp_channels)
        self.LSKModule = LSKModule(feat_channels=self.mlp_channels)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self._init_mod_convs()

    def base_forward(self, inputs):
        out = self.fusion_conv(inputs)
        return out

    def _init_mod_convs(self) -> None:
        """Initialize classification conv layers of the head."""
        # LSKNet
        self.conv_spatial = nn.Conv2d(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            groups=self.feat_channels,
            dilation=1,
        )
        self.conv_squeeze = nn.Conv2d(2, 1, 3, padding=1)
        self.conv_post = nn.Conv2d(self.feat_channels, self.feat_channels, 1)

    def forward_single(
        self, x: Tensor, scale: Scale, stride: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
            predictions of input feature maps.
        """
        # cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        cls_feat = x
        reg_feat = x
        seg_feat = x

        for seg_layer in self.seg_convs:
            seg_feat = seg_layer(seg_feat)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        ################## Changer begins ##################
            
        seg_feat, cls_feat = self.LSKModule(seg_feat, cls_feat)

        seg_feat = self.convs_seg(seg_feat)
        cls_feat = self.convs_cls(cls_feat)
        seg_feat, cls_feat = self.ChannelEx(seg_feat, cls_feat)

        ################### Changer ends ###################

        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        seg_score = self.conv_seg(seg_feat)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness, seg_score

class ChannelAttExchange(nn.Module):
    """
    Channel exchange with dynamic mask prediction using attention.

    Args:
        p (float, optional): Fraction of the features to be exchanged. Defaults to 1/2.
        mlp_channels (int, optional): Number of channels for the MLP. Defaults to 256.
    """

    def __init__(self, p=1/2, mlp_channels=256):
        super().__init__()
        assert 0 <= p <= 1, "p should be between 0 and 1"
        self.p = p
        self.num_exchange_channels = int(mlp_channels * self.p)
        self.mlp = MLP(
            in_channels=self.num_exchange_channels,
            hidden_channels=64,
            out_channels=self.num_exchange_channels,
        )
        self.attention_mask = LSKLayer(mlp_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        N, C, H, W = x1.shape

        x1_mask_logits = self.attention_mask(x1)  # B, C
        x1_mask_logits = self.sigmoid(x1_mask_logits)  # B, C
        x2_mask_logits = self.attention_mask(x2)  # B, C
        x2_mask_logits = self.sigmoid(x2_mask_logits)  # B, C

        x1_topk_values, x1_topk_indices = x1_mask_logits.topk(self.num_exchange_channels, dim=1, largest=True, sorted=False)
        x2_topk_values, x2_topk_indices = x2_mask_logits.topk(self.num_exchange_channels, dim=1, largest=True, sorted=False)

        x1_topk_indices_sorted = torch.sort(x1_topk_indices, dim=1).values
        x2_topk_indices_sorted = torch.sort(x2_topk_indices, dim=1).values

        x1_extracted = x1[torch.arange(N).unsqueeze(-1), x1_topk_indices_sorted]  # B, k, H, W
        x2_extracted = x2[torch.arange(N).unsqueeze(-1), x2_topk_indices_sorted]  # B, k, H, W

        x1_extracted = rearrange(x1_extracted, "b c h w -> b (h w) c")
        x2_extracted = rearrange(x2_extracted, "b c h w -> b (h w) c")

        x1_extracted = self.mlp(x1_extracted)
        x2_extracted = self.mlp(x2_extracted)

        x1_extracted = rearrange(x1_extracted, "b (h w) c -> b c h w", h=H, w=W)
        x2_extracted = rearrange(x2_extracted, "b (h w) c -> b c h w", h=H, w=W)

        out_x1 = x2_extracted
        out_x2 = x1_extracted

        out_x1 = x1.clone()
        out_x2 = x2.clone()

        for i in range(N):
            out_x1[i, x1_topk_indices_sorted[i]] = x2_extracted[i]
            out_x2[i, x2_topk_indices_sorted[i]] = x1_extracted[i]

        return out_x1, out_x2


class LSKModule(nn.Module):
    def __init__(self, feat_channels):
        super(LSKModule, self).__init__()
        self.conv0 = nn.Conv2d(feat_channels, feat_channels, 5, padding=2, groups=feat_channels)
        self.conv_spatial = nn.Conv2d(feat_channels, feat_channels, 7, stride=1, padding=9, groups=feat_channels, dilation=3)
        self.conv1 = nn.Conv2d(feat_channels, feat_channels//2, 1)
        self.conv2 = nn.Conv2d(feat_channels, feat_channels//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(feat_channels//2, feat_channels, 1)

    def forward(self, x1, x2):
        attn1_x1 = self.conv0(x1) # B, C, H, W
        attn2_x1 = self.conv_spatial(attn1_x1) # B, C, H, W
        attn1_x1 = self.conv1(attn1_x1) # B, C//2, H, W
        attn2_x1 = self.conv2(attn2_x1) # B, C//2, H, W

        attn1_x2 = self.conv0(x2) # B, C, H, W
        attn2_x2 = self.conv_spatial(attn1_x2) # B, C, H, W
        attn1_x2 = self.conv1(attn1_x2) # B, C//2, H, W
        attn2_x2 = self.conv2(attn2_x2) # B, C//2, H, W
        
        attn = torch.cat([attn1_x1, attn2_x1], dim=1) # B, C, H, W
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # B, 1, H, W
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # B, 1, H, W
        agg = torch.cat([avg_attn, max_attn], dim=1) # B, 2, H, W
        sig = self.conv_squeeze(agg).sigmoid() # B, 2, H, W
        attn = attn1_x1 * sig[:,0,:,:].unsqueeze(1) + attn2_x1 * sig[:,1,:,:].unsqueeze(1) # B, C//2, H, W
        attn = self.conv(attn) # B, C, H, W
        y1 = attn # B, C, H, W

        attn = torch.cat([attn1_x2, attn2_x2], dim=1) # B, C, H, W
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # B, 1, H, W
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # B, 1, H, W
        agg = torch.cat([avg_attn, max_attn], dim=1) # B, 2, H, W
        sig = self.conv_squeeze(agg).sigmoid() # B, 2, H, W
        attn = attn1_x2 * sig[:,0,:,:].unsqueeze(1) + attn2_x2 * sig[:,1,:,:].unsqueeze(1) # B, C//2, H, W
        attn = self.conv(attn) # B, C, H, W
        y2 = attn # B, C, H, W

        return y1, y2


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LSKLayer(nn.Module):
    def __init__(self, feat_channels):
        super(LSKLayer, self).__init__()
        self.conv0 = nn.Conv2d(feat_channels, feat_channels, 5, padding=2, groups=feat_channels)
        self.conv_spatial = nn.Conv2d(feat_channels, feat_channels, 7, stride=1, padding=9, groups=feat_channels, dilation=3)
        self.conv1 = nn.Conv2d(feat_channels, feat_channels//2, 1)
        self.conv2 = nn.Conv2d(feat_channels, feat_channels//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(feat_channels//2, feat_channels, 1)

    def forward(self, x):
        attn1 = self.conv0(x) # B, C, H, W
        attn2 = self.conv_spatial(attn1) # B, C, H, W
        attn1 = self.conv1(attn1) # B, C//2, H, W
        attn2 = self.conv2(attn2) # B, C//2, H, W
        
        attn = torch.cat([attn1, attn2], dim=1) # B, C, H, W
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # B, 1, H, W
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # B, 1, H, W
        agg = torch.cat([avg_attn, max_attn], dim=1) # B, 2, H, W
        sig = self.conv_squeeze(agg).sigmoid() # B, 2, H, W
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1) # B, C//2, H, W
        attn = self.conv(attn) # B, C, H, W

        y = x * attn # B, C, H, W
        y = y.mean(dim=[2, 3]) # B, C
        return y
