# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DSNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import torch
import os
import detectron2.utils.comm as comm
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
import pdb
from msad import add_msad_config, SLRDetectionCheckpointer
os.environ["NCCL_LL_THRESHOLD"] = "0"

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def modify_checkpoint(self, cfg):
        self.checkpoint = SLRDetectionCheckpointer(self.model, cfg.OUTPUT_DIR, optimizer=self.optimizer, scheduler=self.scheduler)

    def transfer_t_to_s(self):
        if comm.get_world_size()>1:
            # backbone
            self.model.module.S_backbone.load_state_dict(self.model.module.T_backbone.state_dict(), strict=True)
            for param in self.model.module.T_backbone.parameters():
                param.requires_grad = False

            # head
            self.model.module.S_fcos_head.load_state_dict(self.model.module.T_fcos_head.state_dict(), strict=True)
            for param in self.model.module.T_fcos_head.parameters():
                param.requires_grad = False

            # SE layer
            for param in self.model.module.SE.parameters():
                param.requires_grad = False
        else:
            # backbone
            self.model.S_backbone.load_state_dict(self.model.T_backbone.state_dict(), strict=False)
            for param in self.model.T_backbone.parameters():
                param.requires_grad = False

            # head
            self.model.S_fcos_head.load_state_dict(self.model.T_fcos_head.state_dict(), strict=True)
            for param in self.model.T_fcos_head.parameters():
                param.requires_grad = False

            # SE_layer
            for param in self.model.SE.parameters():
                param.requires_grad = False

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_msad_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        SLRDetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # test
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.modify_checkpoint(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.transfer_t_to_s()
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )