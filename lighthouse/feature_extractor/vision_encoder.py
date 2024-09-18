import torch

from lighthouse.feature_extractor.base_encoder import BaseEncoder
from lighthouse.feature_extractor.vision_encoders.resnet152 import ResNet152
from lighthouse.feature_extractor.vision_encoders.clip_v import CLIPVision
from lighthouse.feature_extractor.vision_encoders.slowfast import SlowFast

from lighthouse.frame_loaders.clip_loader import CLIPLoader
from lighthouse.frame_loaders.slowfast_loader import SlowFastLoader

from typing import Optional, Tuple, List, Dict, Any

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

class VisionEncoder(BaseEncoder):

    SLOWFAST_FRAMERATE: float = 30. # 30 is for Slowfast framerate

    def __init__(
        self,
        feature_name: str,
        clip_len: float,
        framerate: float,
        size: int,
        device: str,
        slowfast_path: Optional[str]) -> None:

        self._feature_name: str = feature_name
        self._framerate: float = framerate
        self._clip_len: float = clip_len
        self._size: int = size
        self._device: str = device
        self._slowfast_path: Optional[str] = slowfast_path
        self._frame_loaders = self._select_frame_loader()
        self._visual_encoders = self._select_visual_encoders()

    def _select_frame_loader(self) -> List[Any]:
        frame_loaders: Dict[str, List[Any]] = {
            'resnet_glove' : [CLIPLoader],
            'clip': [CLIPLoader],
            'clip_slowfast': [CLIPLoader, SlowFastLoader],
            'clip_slowfast_pann': [CLIPLoader, SlowFastLoader],
        }

        framerate_dict: Dict[str, List[float]] = {
            'resnet_glove': [self._framerate],
            'clip': [self._framerate],
            'clip_slowfast': [self._framerate, self.SLOWFAST_FRAMERATE],
            'clip_slowfast_pann': [self._framerate, self.SLOWFAST_FRAMERATE],
        }

        loader_instances = [loader(self._clip_len, framerate, self._size, self._device) 
                            for loader, framerate in zip(frame_loaders[self._feature_name], framerate_dict[self._feature_name])]
        return loader_instances
    
    def _select_visual_encoders(self) -> List[Any]:
        visual_encoder_dict: Dict[str, List[Any]] = {
            'resnet_glove': [ResNet152],
            'clip': [CLIPVision],
            'clip_slowfast': [CLIPVision, SlowFast],
            'clip_slowfast_pann': [CLIPVision, SlowFast],
        }

        model_path_dict: Dict[str, List[Optional[str]]] = {
            'resnet_glove': [None],
            'clip': ['ViT-B/32'],
            'clip_slowfast': ['ViT-B/32', self._slowfast_path],
            'clip_slowfast_pann': ['ViT-B/32', self._slowfast_path]
        }

        visual_encoders = [encoder(self._device, model_path)
                           for encoder, model_path in zip(visual_encoder_dict[self._feature_name],
                                                          model_path_dict[self._feature_name])]
        return visual_encoders

    def _trim_shorter_length(self, visual_features):
        min_length = min([x.shape[0] for x in visual_features])
        trimmed_visual_features = [x[:min_length] for x in visual_features]
        return trimmed_visual_features

    def encode(
        self,
        input_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self._frame_loaders) == len(self._visual_encoders), 'the number of frame_loaders and visual_encoders is different.'
        frame_inputs = [loader(input_path) for loader in self._frame_loaders]
        assert not any([item is None for item in frame_inputs]), 'one of the loaders return None object.'
        visual_features = [encoder(frames) for encoder, frames in zip(self._visual_encoders, frame_inputs)]
        concat_features = torch.concat(self._trim_shorter_length(visual_features), dim=-1)
        visual_mask = torch.ones(1, len(concat_features)).to(self._device)
        return concat_features, visual_mask