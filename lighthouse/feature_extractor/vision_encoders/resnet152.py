import math
import torch
import torchvision

from typing import List, Optional

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

class GlobalAvgPool(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])


class Normalize:
    def __init__(
        self,
        mean: List[float],
        std: List[float]) -> None:
        self._mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self._std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self._mean) / (self._std + 1e-8)
        return tensor


class ResNetPreprocessing:
    def __init__(self):
        self._norm = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self._norm(tensor)
        return tensor


class ResNet152:
    def __init__(
        self,
        device: str,
        model_path: Optional[str] = None) -> None:
        assert model_path is None, 'ResNet152 does not use model_path, so should be set None.'
        self._device = device
        resnet_model = torchvision.models.resnet152(pretrained=True)
        self._resnet_extractor = torch.nn.Sequential(
            *list(resnet_model.children())[:-2],
            GlobalAvgPool()).eval().to(device)
        self._preprocess = ResNetPreprocessing()
    
    @torch.no_grad()
    def __call__(
        self,
        video_frames: torch.Tensor,
        bsz: int = 60) -> torch.Tensor:
        video_frames = self._preprocess(video_frames)
        n_frames = len(video_frames)
        n_batch = int(math.ceil(n_frames / bsz))
        video_features = []
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            _video_frames = video_frames[st_idx:ed_idx].to(self._device)
            _video_features = self._resnet_extractor(_video_frames)
            _video_features = torch.nn.functional.normalize(_video_features, dim=-1)
            video_features.append(_video_features)
        video_feature_tensor = torch.cat(video_features, dim=0)
        return video_feature_tensor