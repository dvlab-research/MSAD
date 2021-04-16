import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ConvTranspose2d
from detectron2.structures import ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.postprocessing import detector_postprocess
import fvcore.nn.weight_init as weight_init

from .layers import DFConv2d, IOULoss
from .outputs import FCOSOutputs
from .tower import FCOSHead

from torch.nn import MSELoss as L2Loss

import pdb
import copy
from collections import OrderedDict

__all__ = ["TFCOS"]

INF = 100000000

class SELayer(nn.Module):
    def __init__(self, in_channel=512, output_channel=2, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, output_channel, bias=False),
            nn.Softmax()
        )

        for module in self.fc:
            if isinstance(module, nn.Linear):
                 weight_init.c2_xavier_fill(module)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1, 1, 1)
        return y


@META_ARCH_REGISTRY.register()
class SFCOS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.h_in_features        = cfg.MODEL.FCOS.H_IN_FEATURES
        self.l_in_features        = cfg.MODEL.FCOS.L_IN_FEATURES
        self.num_classes          = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides          = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides              = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius               = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.FCOS.NMS_TH
        self.post_nms_topk_train  = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.FCOS.THRESH_WITH_CTR

        self.T_backbone = build_backbone(cfg)
        self.S_backbone = build_backbone(cfg)

        self.SE = SELayer(reduction=16)

        backbone_shape = self.T_backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.h_in_features]

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        self.T_fcos_head = FCOSHead(cfg, feature_shapes)
        self.S_fcos_head = FCOSHead(cfg, feature_shapes)

        # freeze 
        for param in self.T_backbone.parameters():
            param.requires_grad = False

        for param in self.T_fcos_head.parameters():
            param.requires_grad = False

        for param in self.SE.parameters():
            param.requires_grad = False

        # generate sizes of interest
        self.sizes_of_interest = cfg.MODEL.FCOS.SIZES_OF_INTEREST

        self.joint_weight      = cfg.MODEL.FCOS.JOINT_WEIGHT

        # kd parameters
        self.KD_ALPHA = cfg.MODEL.FCOS.KD_ALPHA
        self.KD_T     = cfg.MODEL.FCOS.KD_TEMPERATURE

        self.count = 0

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Arguments:
            batched_inputs
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        
        # preprocess image, high resolution images, low resolution image
        hr_images, lr_images = self.preprocess_image(batched_inputs)
        # global feature -- [p2, p3, p4, p5, p6, p7]
        hr_features = self.T_backbone(hr_images.tensor)
        lr_features = self.T_backbone(lr_images.tensor)
        st_features = self.S_backbone(lr_images.tensor)

        hr_features = [hr_features[f] for f in self.h_in_features]
        lr_features = [lr_features[f] for f in self.l_in_features]
        st_features = [st_features[f] for f in self.l_in_features]

        fr_features = []
        for hr_feature, lr_feature in zip(hr_features, lr_features):
            fr_feature = torch.cat([hr_feature, lr_feature], dim=1)
            fusion_score = self.SE(fr_feature)
            fr_features.append(fusion_score[:,0].unsqueeze(-1)*hr_feature+fusion_score[:,1].unsqueeze(-1)*lr_feature)
            # fusion_score = self.SE(fr_feature).view(-1)
            # fr_features.append(fusion_score[0]*hr_feature+fusion_score[1]*lr_feature)

        locations = self.compute_locations(hr_features)

        st_logits_pred, st_reg_pred, st_ctrness_pred = self.S_fcos_head(st_features)

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
            hr_gt_instances, lr_gt_instances = self.preprocess_gt(batched_inputs)
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test
            gt_instances = None
            hr_gt_instances = None
            lr_gt_instances = None

        st_outputs = FCOSOutputs(
            hr_images,
            locations,
            st_logits_pred,
            st_reg_pred,
            st_ctrness_pred,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.S_fcos_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            hr_gt_instances
        )

        if self.training:
            st_losses, _ = st_outputs.losses()
            losses = OrderedDict()
            for key, value in st_losses.items():
                losses[key+"_st"] = value

            losses["loss_kd"] = self.kd(fr_features, st_features)
            self.count = self.count+1
            if self.count == 500:
                self.count = 500
            return losses
        else:
            results = st_outputs.predict_proposals()
            processed_results = []
            height = batched_inputs[0].get("height", hr_images.image_sizes[0][0])
            width = batched_inputs[0].get("width", hr_images.image_sizes[0][1])
            r = detector_postprocess(results[0], height, width)
            processed_results = [{"instances": r}]
            return processed_results

    def kd(self, t_features, s_features):
        d_fpn_loss = 0
        for index, (t_feature, s_feature) in enumerate(zip(t_features, s_features)):
            tt_feature = (t_feature / self.KD_T)
            ss_feature = (s_feature / self.KD_T)
            d_fpn_loss += F.l1_loss(ss_feature,tt_feature).mean() * self.KD_ALPHA * self.KD_T * self.KD_T * min(1.0, self.count/500.0)
        return d_fpn_loss

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device).float() for x in batched_inputs]
        hr_images = copy.deepcopy(images)
        hr_images = [self.normalizer(x) for x in hr_images]
        hr_images = ImageList.from_tensors(hr_images, 256)
        lr_images_tensor = F.interpolate(hr_images.tensor, scale_factor=0.5, mode="nearest")
        lr_images_sizes = [(int(image_h/2), int(image_w/2)) for image_h, image_w in hr_images.image_sizes]
        lr_images = ImageList(lr_images_tensor, lr_images_sizes)
        return hr_images, lr_images

    def preprocess_gt(self, batched_inputs):
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        hr_gt_instances = copy.deepcopy(gt_instances)
        lr_gt_instances = copy.deepcopy(gt_instances)

        for lr_gt_instance in lr_gt_instances:
            lr_gt_instance.gt_boxes.tensor *= 0.5
            h, w = lr_gt_instance._image_size
            lr_gt_instance._image_size = (int(h*0.5), int(w*0.5))

        return hr_gt_instances, lr_gt_instances