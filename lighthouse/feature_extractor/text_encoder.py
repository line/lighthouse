import torch

from lighthouse.feature_extractor.base_encoder import BaseEncoder
from lighthouse.feature_extractor.text_encoders.clip_t import CLIPText
from lighthouse.feature_extractor.text_encoders.glove import GloVe
from lighthouse.feature_extractor.text_encoders.clap_t import CLAPText

from typing import Tuple

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

class TextEncoder(BaseEncoder):
    def __init__(
        self,
        feature_name: str,
        device: str) -> None:
        self._feature_name: str = feature_name
        self._device: str = device
        self._text_encoders = self._select_text_encoders()

    def _select_text_encoders(self):
        text_encoders = {
            'resnet_glove': [GloVe],
            'clip': [CLIPText],
            'clip_slowfast': [CLIPText],
            'clip_slowfast_pann': [CLIPText],
            'clap': [CLAPText],
        }

        model_path_dict = {
            'resnet_glove': ['glove.6B.300d'],
            'clip': ['ViT-B/32'],
            'clip_slowfast': ['ViT-B/32'],
            'clip_slowfast_pann': ['ViT-B/32'],
            'clap': ['2023'],
        }

        text_encoders = [encoder(self._device, model_path)
                         for encoder, model_path in zip(text_encoders[self._feature_name], model_path_dict[self._feature_name])]
        return text_encoders

    def encode(
        self,
        query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = [encoder(query) for encoder in self._text_encoders]
        text_features = torch.cat([o[0] for o in outputs])
        text_masks = torch.cat([o[1] for o in outputs])
        return text_features, text_masks
