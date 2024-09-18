import math
import torch
import ffmpeg
import numpy as np
import random

from typing import Optional, Tuple, List
from lighthouse.frame_loaders.base_loader import BaseLoader, convert_to_float

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


class SlowFastLoader(BaseLoader):
    def __init__(
        self,
        clip_len: float,
        framerate: int, # override = 30 for PySlowFast implementation
        size: int,
        device: str,
        centercrop: bool = True) -> None:
        super().__init__(clip_len, framerate, size, device, centercrop)
        self._preprocess = Preprocessing('3d', device=device, target_fps=framerate,
                                        size=224, clip_len=clip_len,
                                        padding_mode='tile', min_num_clips=1)

    def __call__(
        self,
        video_path: str) -> Optional[torch.Tensor]:
        
        info = self._video_info(video_path)
        if info is None:
            return None

        h, w = info["height"], info["width"]
        height, width = self._output_dim(h, w)
        
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=self._framerate)
            .filter('scale', width, height)
        )

        if self._centercrop:
            x = int((width - self._size) / 2.0)
            y = int((height - self._size) / 2.0)
            cmd = cmd.crop(x, y, self._size, self._size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )

        if self._centercrop and isinstance(self._size, int):
            height, width = self._size, self._size
        video = np.frombuffer(out, np.uint8).reshape(
            [-1, height, width, 3])
        video_tensor = torch.from_numpy(video)
        video_tensor = self._preprocess(video_tensor, info)
        return video_tensor


class Preprocessing:
    def __init__(
        self,
        type : str,
        device: str,
        target_fps: int = 16,
        size: int = 112,
        clip_len: float = 2.,
        padding_mode: str = 'tile',
        min_num_clips: int = 1):
        self._type = type
        self._device = device
        self._norm = Normalize(mean=[0.45, 0.45, 0.45], 
                              std=[0.225, 0.225, 0.225],
                              device=device)
        self._target_fps = target_fps
        self._num_frames = 32
        self._sampling_rate = 2
        self._size = size
        self._clip_len = clip_len
        self._padding_mode = padding_mode
        self._min_num_clips = min_num_clips


    def _pad_frames(self, tensor, value=0):
        n = self._target_fps - len(tensor) % self._target_fps
        if n == self._target_fps:
            return tensor
        if self._padding_mode == "constant":
            z = torch.ones(int(n), tensor.shape[1], tensor.shape[2], tensor.shape[3], dtype=torch.uint8)
            z *= value
            return torch.cat((tensor, z), 0)
        elif self._padding_mode == "tile":
            z = torch.cat(int(n) * [tensor[-1:, :, :, :]])
            return torch.cat((tensor, z), 0)
        else:
            raise NotImplementedError(
                f'Mode {self._padding_mode} not implemented in _pad_frames.')

    def _pad_clips(self, tensor):
        n = self._clip_len - len(tensor) % self._clip_len
        
        if n == self._clip_len:
            return tensor

        z = torch.cat(int(n) * [tensor[-1:, :, :, :, :]])
        return torch.cat((tensor, z), 0)

    def __call__(self, tensor, info):
        target_fps = int(self._target_fps)
        tensor = self._pad_frames(tensor, self._padding_mode)
        tensor = tensor.view(-1, target_fps, self._size, self._size, 3)
        tensor = self._pad_clips(tensor)
        clip_len = convert_to_float(self._clip_len)
        clips = tensor.view(
                -1, int(clip_len * target_fps), self._size, self._size, 3)
        
        try:
            duration = info["duration"]
            if duration > 0:
                num_clips = int(math.ceil(duration / clip_len))
                clips = clips[:num_clips]
        except Exception:
            raise RuntimeError("Duration not available.")
        
        num_clips = len(clips)
        if num_clips < self._min_num_clips:
            clips = clips.view(
                self._min_num_clips, -1, self._size, self._size, 3)
        
        fps = info["fps"]
        start_idx, end_idx = get_start_end_idx(
            video_size = clips.shape[1],
            clip_size = self._num_frames * self._sampling_rate * fps / self._target_fps,
            clip_idx = 0,
            num_clips = 1,
        )
        clips = temporal_sampling(
            clips, start_idx, end_idx, self._num_frames)
        return clips


class Normalize:
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

def get_start_end_idx(
    video_size: int,
    clip_size: int,
    clip_idx: int,
    num_clips: int) -> Tuple[int, int]:
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        start_idx = int(random.uniform(0, delta))
    else:
        start_idx = int(delta * clip_idx / num_clips)
    end_idx = int(start_idx + clip_size - 1)
    return start_idx, end_idx


def temporal_sampling(
    frames: torch.Tensor,
    start_idx: int,
    end_idx: int,
    num_samples: int) -> torch.Tensor:
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `batch_size` x `num video frames` x `height` x `width` x `channel`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames,
            dimension is
            `batch_size` x `num clip frames`` x `height` x `width` x `channel.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[1] - 1).long()
    frames = torch.index_select(frames, 1, index)
    return frames