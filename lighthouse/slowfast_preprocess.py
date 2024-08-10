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

MIT License

Copyright (c) 2020 Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch as th
import random
import numpy as np
import math


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
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
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def temporal_sampling(frames, start_idx, end_idx, num_samples):
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
    index = th.linspace(start_idx, end_idx, num_samples)
    index = th.clamp(index, 0, frames.shape[1] - 1).long()
    frames = th.index_select(frames, 1, index)
    return frames


class Normalize(object):

    def __init__(self, mean, std, device):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1, 1).float().to(device)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1, 1).float().to(device)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):
    def __init__(self, type, device, target_fps=16, size=112,
                 clip_len=2, padding_mode='tile', min_num_clips=1):
        self.type = type
        self.device = device
        self.norm = Normalize(
            mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], device=device)
        self.target_fps = target_fps
        self.num_frames = 32
        self.sampling_rate = 2
        self.size = size
        self.clip_len = clip_len
        self.padding_mode = padding_mode
        self.min_num_clips = min_num_clips

    def _tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = th.LongTensor(
            np.concatenate(
                [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return th.index_select(a, dim, order_index)

    def _pad_frames(self, tensor, mode='tile', value=0):
        # print(f"Target fps {self.target_fps} not satisfied, padding....")
        n = self.target_fps - len(tensor) % self.target_fps
        if n == self.target_fps:
            return tensor
        if mode == "constant":
            z = th.ones(n, tensor.shape[1], tensor.shape[2], tensor.shape[3],
                        dtype=th.uint8)
            z *= value
            return th.cat((tensor, z), 0)
        elif mode == "tile":
            z = th.cat(n * [tensor[-1:, :, :, :]])
            return th.cat((tensor, z), 0)
        else:
            raise NotImplementedError(
                f'Mode {mode} not implemented in _pad_frames.')

    def _pad_clips(self, tensor, mode='tile', value=0):
        # print(f"clip length {self.clip_len} not satisfied, padding....")
        clip_len = self.clip_len
        n = clip_len - len(tensor) % clip_len
        if n == clip_len:
            return tensor
        z = th.cat(int(n) * [tensor[-1:, :, :, :, :]])
        return th.cat((tensor, z), 0)

    def __call__(self, tensor, info):
        tensor = self._pad_frames(tensor, self.padding_mode)
        # (duration [in seconds], # frames, height, width, channel)
        tensor = tensor.view(-1, self.target_fps, self.size, self.size, 3)
        # (# of clips, # of clip frames, height, width, channel)
        tensor = self._pad_clips(tensor, self.padding_mode)
        clip_len = convert_to_float(self.clip_len)
        clips = tensor.view(
                -1, int(clip_len*self.target_fps), self.size, self.size, 3)
        try:
            duration = info["duration"]
            if duration > 0:
                num_clips = int(math.ceil(duration/clip_len))
                clips = clips[:num_clips]
        except Exception:
            print("Duration not available...")
        num_clips = len(clips)
        if num_clips < self.min_num_clips:
            clips = clips.view(
                self.min_num_clips, -1, self.size, self.size, 3)
        # assert th.equal(clips[1, 0, :, :, :],
        #                 tensor[clip_len, 0, :, :, :])
        fps = info["fps"]  # .item()
        start_idx, end_idx = get_start_end_idx(
            clips.shape[1],
            self.num_frames * self.sampling_rate * fps / self.target_fps,
            0,
            1,
        )
        # Perform temporal sampling from the decoded video.
        clips = temporal_sampling(
            clips, start_idx, end_idx, self.num_frames)
        # B T H W C
        # clips = clips.transpose(1, 2)
        return clips