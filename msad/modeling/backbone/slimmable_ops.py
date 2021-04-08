import torch.nn as nn
from collections import defaultdict
from detectron2.layers import get_norm
from detectron2.layers.batch_norm import FrozenBatchNorm2d
import pdb

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list, cfg=None):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        norm = cfg.MODEL.SLRESNETS.NORM
        for i in num_features_list:
            bns.append(get_norm(norm, i))
        self.bn = nn.ModuleList(bns)
        self.ignore_model_profiling = True
        self.width_mult_list = cfg.MODEL.SLRESNETS.WIDTH_MULT_LIST
        self.width_mult = cfg.MODEL.SLRESNETS.WIDTH_MULT

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y

    def freeze(self):
        for bn in self.bn:
            FrozenBatchNorm2d.convert_frozen_batchnorm(bn)
        return self

class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True, cfg=None):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        # self.width_mult = max(FLAGS["width_mult_list"])
        self.width_mult_list = cfg.MODEL.SLRESNETS.WIDTH_MULT_LIST
        self.width_mult = cfg.MODEL.SLRESNETS.WIDTH_MULT

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True, cfg=None):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult_list = cfg.MODEL.SLRESNETS.WIDTH_MULT_LIST
        self.width_mult = cfg.MODEL.SLRESNETS.WIDTH_MULT
        # self.width_mult = FLAGS["width_mult"]
        # self.width_mult = max(FLAGS["width_mult_list"])

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False