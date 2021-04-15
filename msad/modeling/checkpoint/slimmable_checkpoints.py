# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
from fvcore.common.checkpoint import Checkpointer, _strip_prefix_if_present, _IncompatibleKeys
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
import pdb
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

class SLRDetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        if "0.5_0.75_1.0" in filename:
            loaded.update({"slimmable": True})
        return loaded

    def _load_model(self, checkpoint):
        # pdb.set_trace()
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        if checkpoint.get("slimmable", False):
            incompatible = self._load_model_slimmable(checkpoint)
        else:
            incompatible = super()._load_model(checkpoint)
        if incompatible is None:  # support older versions of fvcore
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

    def _load_model_slimmable(self, checkpoint):
        checkpoint_state_dict = checkpoint.pop("model")
        _strip_prefix_if_present(checkpoint_state_dict, "module.")
        # to new checkpoint
        new_checkpoint_state_dict = {}
        for k in list(checkpoint_state_dict.keys()):
            new_checkpoint_state_dict["T_backbone.bottom_up.{}".format(k)] = checkpoint_state_dict[k]
        # import pdb.set_trace()

        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        # pdb.set_trace()
        for k in list(new_checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(new_checkpoint_state_dict[k].shape)
                # import pdb
                # pdb.set_trace()
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    new_checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.model.load_state_dict(new_checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

