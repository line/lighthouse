import torch
import math

from typing import List, Optional
from lighthouse.feature_extractor.vision_encoders.slowfast_model.model_loader import slowfast_model_loader

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

class Normalize(object):
    def __init__(
        self,
        mean: List[float],
        std: List[float]) -> None:
        self._mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self._std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self._mean) / (self._std + 1e-8)
        return tensor


class SlowFastNormalize(object):
    def __init__(
        self,
        mean: List[float],
        std: List[float],
        device: str) -> None:
        self._mean = torch.FloatTensor(mean).view(1, 3, 1, 1, 1).float().to(device)
        self._std = torch.FloatTensor(std).view(1, 3, 1, 1, 1).float().to(device)

    def __call__(self, tensor):
        tensor = (tensor - self._mean) / (self._std + 1e-8)
        return tensor


class SlowFast:

    SLOWFAST_FEATURE_DIM = 2304

    def __init__(
        self,
        device: str,
        model_path: Optional[str]) -> None:
        assert model_path is not None, 'Slowfast use model_path, so should be set but None.'
        self._device = device
        self._slowfast_norm = SlowFastNormalize(mean=[0.45, 0.45, 0.45], 
                                                std=[0.225, 0.225, 0.225], 
                                                device=device)
        self._slowfast_extractor = slowfast_model_loader(model_path, device)

    def _pack_pathway_output(
        self,
        frames: torch.Tensor) -> List[torch.Tensor]:
        """
        Prepare output as a list of tensors. Each tensor corresponding to a
        unique pathway.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is
                `batch` x `channel` x `num frames` x `height` x `width`.
        Returns:
            frame_list (list): list of tensors with the dimension of
                `batch` x `channel` x `num frames` x `height` x `width`.
        """
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // 4
            ).long().to(self._device),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
    
    @torch.no_grad()
    def __call__(
        self,
        slowfast_frames: torch.Tensor,
        bsz: int = 45):
        n_chunk = len(slowfast_frames)
        features = torch.zeros([n_chunk, self.SLOWFAST_FEATURE_DIM],
                                device=self._device, dtype=torch.float16)
        n_batch = int(math.ceil(n_chunk / bsz))
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            fast_clip = slowfast_frames[st_idx:ed_idx].float().to(self._device)
            fast_clip = fast_clip.permute(0, 4, 1, 2, 3)
            fast_clip = fast_clip/255.
            fast_clip = self._slowfast_norm(fast_clip)
            inputs = self._pack_pathway_output(fast_clip)
            batch_features = self._slowfast_extractor(inputs)
            features[st_idx:ed_idx] = batch_features.half()
        return features