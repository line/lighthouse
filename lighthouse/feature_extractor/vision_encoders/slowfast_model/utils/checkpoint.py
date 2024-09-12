import os
import pickle
import torch

from collections import OrderedDict
from lighthouse.feature_extractor.vision_encoders.slowfast_model.utils.c2_model_loading import get_name_convert_func

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
"""Functions that handle saving and loading of checkpoints."""


def inflate_weight(state_dict_2d, state_dict_3d):
    """
    Inflate 2D model weights in state_dict_2d to the 3D model weights in
    state_dict_3d. The details can be found in:
    Joao Carreira, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    Args:
        state_dict_2d (OrderedDict): a dict of parameters from a 2D model.
        state_dict_3d (OrderedDict): a dict of parameters from a 3D model.
    Returns:
        state_dict_inflated (OrderedDict): a dict of inflated parameters.
    """
    state_dict_inflated = OrderedDict()
    for k, v2d in state_dict_2d.items():
        assert k in state_dict_3d.keys()
        v3d = state_dict_3d[k]
        # Inflate the weight of 2D conv to 3D conv.
        if len(v2d.shape) == 4 and len(v3d.shape) == 5:
            # Dimension need to be match.
            assert v2d.shape[-2:] == v3d.shape[-2:]
            assert v2d.shape[:2] == v3d.shape[:2]
            v3d = (
                v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            )
        if v2d.shape == v3d.shape:
            v3d = v2d
        state_dict_inflated[k] = v3d.clone()
    return state_dict_inflated


def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    inflation=False,
    convert_from_caffe2=False,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert os.path.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    if convert_from_caffe2:
        with open(path_to_checkpoint, "rb") as f:
            caffe2_checkpoint = pickle.load(f, encoding="latin1")
        state_dict = OrderedDict()
        name_convert_func = get_name_convert_func()
        for key in caffe2_checkpoint["blobs"].keys():
            converted_key = name_convert_func(key)
            if converted_key in ms.state_dict():
                if caffe2_checkpoint["blobs"][key].shape == tuple(
                    ms.state_dict()[converted_key].shape
                ):
                    state_dict[converted_key] = torch.tensor(
                        caffe2_checkpoint["blobs"][key]
                    ).clone()
            else:
                assert any(
                    prefix in key for prefix in ["momentum", "lr", "model_iter"]
                ), "{} can not be converted, got {}".format(key, converted_key)
        ms.load_state_dict(state_dict, strict=False)
        epoch = -1
    else:
        # Load the checkpoint on CPU to avoid GPU mem spike.
        checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
        if inflation:
            # Try to inflate the model.
            model_state_dict_3d = (
                model.module.state_dict()
                if data_parallel
                else model.state_dict()
            )
            inflated_model_dict = inflate_weight(
                checkpoint["model_state"], model_state_dict_3d
            )
            ms.load_state_dict(inflated_model_dict, strict=False)
        else:
            ms.load_state_dict(checkpoint["model_state"])
            # Load the optimizer state (commonly not done when fine-tuning)
            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "epoch" in checkpoint.keys():
            epoch = checkpoint["epoch"]
        else:
            epoch = -1
    return epoch
