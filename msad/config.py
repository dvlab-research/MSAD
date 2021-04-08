# -*- coding: utf-8 -*-
# Based on FCOS
from detectron2.config import CfgNode as CN

def add_msad_config(cfg):
    """
    Add config for msad.
    """
    # slimmable ResNets
    cfg.MODEL.SLRESNETS = CN()
    cfg.MODEL.SLRESNETS.NORM  = "BN"
    cfg.MODEL.SLRESNETS.DEPTH = 50
    cfg.MODEL.SLRESNETS.WIDTH_MULT_LIST = [0.25, 0.50, 0.75, 1.0]
    cfg.MODEL.SLRESNETS.WIDTH_MULT      = 1.0

    # FCOS Parameters
    cfg.MODEL.FCOS = CN()

    # Anchor parameters
    cfg.MODEL.FCOS.H_IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.L_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.FCOS.NUM_CLASSES = 80
    # cfg.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
    cfg.MODEL.FCOS.SIZES_OF_INTEREST = [[-1, 64], [64,128], [128,256], [256,512], [512, 100000000]]
    # cfg.MODEL.FCOS.SIZES_OF_INTEREST = [[-1, 32], [32,64], [64,128], [128,256], [256, 100000000]]

    # tower
    cfg.MODEL.FCOS.NUM_CLS_CONVS   = 4
    cfg.MODEL.FCOS.NUM_BOX_CONVS   = 4
    cfg.MODEL.FCOS.NUM_SHARE_CONVS = 0
    cfg.MODEL.FCOS.CENTER_SAMPLE   = True
    cfg.MODEL.FCOS.POS_RADIUS      = 1.5
    cfg.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    cfg.MODEL.FCOS.USE_RELU = True
    cfg.MODEL.FCOS.USE_DEFORMABLE = False
    cfg.MODEL.FCOS.USE_SCALE  = True
    cfg.MODEL.FCOS.TOP_LEVELS = 2
    cfg.MODEL.FCOS.NORM = "GN"

   # loss
    cfg.MODEL.FCOS.PRIOR_PROB    = 0.01
    cfg.MODEL.FCOS.LOSS_ALPHA    = 0.25
    cfg.MODEL.FCOS.LOSS_GAMMA    = 2.0
    cfg.MODEL.FCOS.CENTER_SAMPLE = True

    # teacher unique parameters
    cfg.MODEL.FCOS.JOINT_WEIGHT  = 0.4

    # student unique parameters
    cfg.MODEL.FCOS.KD_ALPHA = 0.2
    cfg.MODEL.FCOS.KD_TEMPERATURE = 3.0

    # inference
    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN  = 0.05
    cfg.MODEL.FCOS.INFERENCE_TH_TEST   = 0.05
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN  = 1000
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST   = 1000
    cfg.MODEL.FCOS.NMS_TH              = 0.6
    cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    cfg.MODEL.FCOS.POST_NMS_TOPK_TEST  = 100
    cfg.MODEL.FCOS.THRESH_WITH_CTR     = False



