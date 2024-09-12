import torch

from lighthouse.feature_extractor.base_encoder import BaseEncoder
from lighthouse.feature_extractor.text_encoders.clip_t import CLIPText
from lighthouse.feature_extractor.text_encoders.glove import GloVe

from typing import Tuple

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
        }

        model_path_dict = {
            'resnet_glove': ['glove.6B.300d'],
            'clip': ['ViT-B/32'],
            'clip_slowfast': ['ViT-B/32'],
            'clip_slowfast_pann': ['ViT-B/32'],
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