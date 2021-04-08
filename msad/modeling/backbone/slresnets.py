import torch.nn as nn
import math
import torch
from collections import defaultdict

import pdb
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear

from detectron2.layers import CNNBlockBase, ShapeSpec

__all__ = [
    "Block",
    "SL_RESNET",
    "build_sresnet_backbone",
]

class Block(nn.Module):
    def __init__(self, inp, outp, stride, cfg=None):
        super(Block, self).__init__()
        assert stride in [1, 2]

        midp = [i // 4 for i in outp]
        layers = [
            SlimmableConv2d(inp, midp, 1, 1, 0, bias=False, cfg=cfg),
            SwitchableBatchNorm2d(midp, cfg=cfg),
            nn.ReLU(inplace=True),

            SlimmableConv2d(midp, midp, 3, stride, 1, bias=False, cfg=cfg),
            SwitchableBatchNorm2d(midp, cfg=cfg),
            nn.ReLU(inplace=True),

            SlimmableConv2d(midp, outp, 1, 1, 0, bias=False, cfg=cfg),
            SwitchableBatchNorm2d(outp, cfg=cfg),
        ]
        self.body = nn.Sequential(*layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                SlimmableConv2d(inp, outp, 1, stride=stride, bias=False, cfg=cfg),
                SwitchableBatchNorm2d(outp, cfg=cfg),
            )
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
            res += self.shortcut(x)
        res = self.post_relu(res)
        return res

class SL_RESNET(Backbone):
    def __init__(self, cfg):
        super(SL_RESNET, self).__init__()

        norm            = cfg.MODEL.SLRESNETS.NORM
        depth           = cfg.MODEL.SLRESNETS.DEPTH
        width_mult_list = cfg.MODEL.SLRESNETS.WIDTH_MULT_LIST
        self.width_mult = cfg.MODEL.SLRESNETS.WIDTH_MULT

        block_setting_dict = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],}
        self.block_setting = block_setting_dict[depth]

        self.features = []
        # setting

        feats = [64, 128, 256, 512]
        channels = [int(64 * width_mult) for width_mult in width_mult_list]

        self.features.append(nn.Sequential(SlimmableConv2d([3 for _ in range(len(channels))], channels, 7, 2, 3, bias=False, cfg=cfg), 
                                           SwitchableBatchNorm2d(channels, cfg=cfg), 
                                           nn.ReLU(inplace=True),
                                           nn.MaxPool2d(3, 2, 1),))

        # body
        for stage_id, n in enumerate(self.block_setting):
            outp = [int(feats[stage_id] * width_mult * 4) for width_mult in width_mult_list]
            for i in range(n):
                if i == 0 and stage_id != 0:
                    self.features.append(Block(channels, outp, 2, cfg=cfg))
                else:
                    self.features.append(Block(channels, outp, 1, cfg=cfg))
                channels = outp

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

    def output_shape(self):
        info = {"res1": (int(64*self.width_mult), 2), "res2": (int(256*self.width_mult), 4), "res3": (int(512*self.width_mult), 8), "res4": (int(1024*self.width_mult), 16), "res5": (int(2048*self.width_mult), 32)}
        return {name: ShapeSpec(channels=info[name][0], stride=info[name][1]) for name in info.keys()}

    def forward(self, x):
        outputs = {}
        # head
        x = self.features[0][0:3](x)
        outputs["res1"] = x
        x = self.features[0][3](x)
        # c2
        x = self.features[1:1+self.block_setting[0]](x)
        outputs["res2"] = x
        # c3
        x = self.features[1+self.block_setting[0]:1+self.block_setting[0]+self.block_setting[1]](x)
        outputs["res3"] = x
        # c4
        x = self.features[1+self.block_setting[0]+self.block_setting[1]:1+self.block_setting[0]+self.block_setting[1]+self.block_setting[2]](x)
        outputs["res4"] = x
        # c5
        x = self.features[1+self.block_setting[0]+self.block_setting[1]+self.block_setting[2]:1+self.block_setting[0]+self.block_setting[1]+self.block_setting[2]+self.block_setting[3]](x)
        outputs["res5"] = x

        return outputs

    def init_weights(self, pretrained):
        # pretrain_dict = torch.load("/data/qilu/projects/pretrained_model/s_resnet50_0.25_0.5_0.75_1.0.pt", map_location=lambda storage, loc: storage)
        checkpoint = torch.load("/data/qilu/projects/pretrained_model/s_resnet50_0.25_0.5_0.75_1.0.pt")["model"]
        
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
def build_slresnet_backbone(cfg):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    s_resnet = SL_RESNET(cfg)
    return SL_RESNET(cfg).freeze(freeze_at)





