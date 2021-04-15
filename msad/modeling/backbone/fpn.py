import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn
import pdb

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7
from .slresnets import build_slresnet_backbone
from .slmobilenetv2 import build_slmobilev2_backbone


@BACKBONE_REGISTRY.register()
def build_slresnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up        = build_slresnet_backbone(cfg)
    in_features      = cfg.MODEL.FPN.IN_FEATURES
    out_channels     = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_slmobilev2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up        = build_slmobilev2_backbone(cfg)
    in_features      = cfg.MODEL.FPN.IN_FEATURES
    out_channels     = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE
    )
    return backbone