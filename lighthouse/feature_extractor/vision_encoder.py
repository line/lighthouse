import torch

from lighthouse.feature_extractor.base_encoder import BaseEncoder
from lighthouse.feature_extractor.vision_encoders.resnet152 import ResNet152
from lighthouse.feature_extractor.vision_encoders.clip_v import CLIPVision
from lighthouse.feature_extractor.vision_encoders.slowfast import SlowFast

from lighthouse.frame_loaders.clip_loader import CLIPLoader
from lighthouse.frame_loaders.slowfast_loader import SlowFastLoader

from typing import Optional


class VisionEncoder(BaseEncoder):
    def __init__(
        self,
        feature_name: str,
        framerate: float,
        size: int,
        device: str,
        slowfast_path: Optional[str]) -> None:

        self._feature_name: str = feature_name
        self._framerate: float = framerate
        self._size: int = size
        self._device: str = device
        self._slowfast_path: Optional[str] = slowfast_path
        self._frame_loaders = self._select_frame_loader()
        self._visual_encoders = self._select_visual_encoders()

    def _select_frame_loader(self):
        frame_loaders = {
            'resnet_glove' : [CLIPLoader],
            'clip': [CLIPLoader],
            'clip_slowfast': [CLIPLoader, SlowFastLoader],
            'clip_slowfast_pann': [CLIPLoader, SlowFastLoader],
        }
        loader_instances = [loader() for loader in frame_loaders[self._feature_name]]
        return loader_instances
    
    def _select_visual_encoders(self):
        visual_encoders = {
            'resnet_glove': [ResNet152],
            'clip': [CLIPVision],
            'clip_slowfast': [CLIPVision, SlowFast],
            'clip_slowfast_pann': [CLIPVision, SlowFast],
        }
        visual_encoders = [encoder() for encoder in visual_encoders[self._feature_name]]
        return visual_encoders

    def encode(self,
        input_path: str) -> torch.Tensor:
        assert len(self._frame_loaders) == len(self._visual_encoders)
        frame_inputs = [loader(input_path) for loader in self._frame_loaders]
        visual_features = [encoder(frames) for encoder, frames in zip(self._visual_encoders, frame_inputs)]
        # TODO: concat
        return visual_features