import math
import torch.nn as nn

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import get_norm

from .slimmable_ops import SwitchableBatchNorm2d, SlimmableConv2d
from .slimmable_ops import make_divisible

from detectron2.layers import CNNBlockBase, ShapeSpec
import pdb


class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio, cfg=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = [i * expand_ratio for i in inp]
        if expand_ratio != 1:
            layers += [
                SlimmableConv2d(inp, expand_inp, 1, 1, 0, bias=False, cfg=cfg),
                SwitchableBatchNorm2d(expand_inp, cfg=cfg),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SlimmableConv2d(expand_inp, expand_inp, 3, stride, 1, groups_list=expand_inp, bias=False, cfg=cfg),
            SwitchableBatchNorm2d(expand_inp, cfg=cfg),
            nn.ReLU6(inplace=True),
            SlimmableConv2d(expand_inp, outp, 1, 1, 0, bias=False, cfg=cfg),
            SwitchableBatchNorm2d(outp, cfg=cfg),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class SL_MOBILETV2(Backbone):
    def __init__(self, cfg):
        super(SL_MOBILETV2, self).__init__()

        norm            = cfg.MODEL.SLRESNETS.NORM
        self.width_mult_list = cfg.MODEL.SLRESNETS.WIDTH_MULT_LIST
        self.width_mult = cfg.MODEL.SLRESNETS.WIDTH_MULT
        self.pretrained_path = cfg.MODEL.WEIGHTS

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2], # 4
            [6, 32, 3, 2], # 8
            [6, 64, 4, 2], # 16
            [6, 96, 3, 1],
            [6, 160, 3, 2], # 32
            [6, 320, 1, 1],
        ]

        self.features = []

        # head
        channels = [make_divisible(32 * width_mult) for width_mult in self.width_mult_list]
        self.outp = make_divisible(1280 * max(self.width_mult_list)) if max(self.width_mult_list) > 1.0 else 1280
        
        first_stride = 2
        self.features.append(nn.Sequential(SlimmableConv2d([3 for _ in range(len(channels))], channels, 3, first_stride, 1, bias=False, cfg=cfg), 
                                           SwitchableBatchNorm2d(channels,cfg=cfg),
                                           nn.ReLU6(inplace=True)))

        # body
        for t, c, n, s in self.block_setting:
            outp = [make_divisible(c * width_mult) for width_mult in self.width_mult_list]
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(channels, outp, s, t, cfg=cfg))
                else:
                    self.features.append(InvertedResidual(channels, outp, 1, t, cfg=cfg))
                channels = outp

        # tail
        self.features.append(nn.Sequential(SlimmableConv2d(channels, [self.outp for _ in range(len(channels))], 1, 1, 0, bias=False, cfg=cfg),
                get_norm(norm, self.outp),
                nn.ReLU6(inplace=True),
            )
        )

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

    def output_shape(self):
        info = {"res1": (int(32*self.width_mult), 2), "res2": (int(24*self.width_mult), 4), "res3": (int(32*self.width_mult), 8), "res4": (int(96*self.width_mult), 16), "res5": (int(1280*self.width_mult), 32)}
        return {name: ShapeSpec(channels=info[name][0], stride=info[name][1]) for name in info.keys()}


    def forward(self, input_):
        outputs = {}
        # head
        x = self.features[0](input_)
        outputs["res1"] = x
        # c2
        x = self.features[1:1+self.block_setting[0][2]+self.block_setting[1][2]](x)
        outputs["res2"] = x
        # c3
        x = self.features[1+self.block_setting[0][2]+self.block_setting[1][2]:1+self.block_setting[0][2]+self.block_setting[1][2]+self.block_setting[2][2]](x)
        outputs["res3"] = x
        # c4
        x = self.features[1+self.block_setting[0][2]+self.block_setting[1][2]+self.block_setting[2][2]:1+self.block_setting[0][2]+self.block_setting[1][2]+self.block_setting[2][2]+self.block_setting[3][2]+self.block_setting[4][2]](x)
        outputs["res4"] = x
        # c5
        x = self.features[1+self.block_setting[0][2]+self.block_setting[1][2]+self.block_setting[2][2]+self.block_setting[3][2]+self.block_setting[4][2]:](x)
        outputs["res5"] = x
        
        return outputs

    def init_weights(self, pretrained):
        checkpoint = torch.load(self.pretrained_path)["model"]
        
        new_keys = list(self.state_dict().keys())
        old_keys = list(checkpoint.keys())

        new_checkpoint = {}
        used_keys = []
        
        for old_key in old_keys:
            if old_key.split(".",1)[1] in new_keys:
                new_checkpoint[old_key.split(".",1)[1]] = checkpoint[old_key]
                used_keys.append(old_key.split(".",1)[1])

        self.load_state_dict(new_checkpoint, strict=True)

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            upper = sum(self.block_setting[0:freeze_at-1])
            for index, modules in enumerate(self.features[0:1+upper]):
                if index==0:
                    for module in modules:
                        if isinstance(module, SlimmableConv2d) or isinstance(module, SwitchableBatchNorm2d):
                            module.freeze()
                else:
                    for module in modules.body:
                        if isinstance(module, SlimmableConv2d) or isinstance(module, SwitchableBatchNorm2d):
                            module.freeze()
                    if hasattr(modules, "shortcut"):
                        for module in modules.shortcut:
                            if isinstance(module, SlimmableConv2d) or isinstance(module, SwitchableBatchNorm2d):
                                module.freeze()
        elif freeze_at == 0:
            self.features[0][0].freeze()
            self.features[0][1].freeze()

        return self

@BACKBONE_REGISTRY.register()
def build_slmobilev2_backbone(cfg):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    s_resnet = SL_MOBILETV2(cfg)
    return SL_MOBILETV2(cfg).freeze(freeze_at)