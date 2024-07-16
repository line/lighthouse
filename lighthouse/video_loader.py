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

Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei

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


import math
import torch
import ffmpeg
import numpy as np

from lighthouse.slowfast_preprocess import Preprocessing

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

class VideoLoader:
    """
    Pytorch video loader.
    Copied and modified from:
    https://github.com/linjieli222/HERO_Video_Feature_Extractor/blob/main/clip/video_loader.py
    """
    def __init__(
            self,
            framerate=1/2,
            size=224,
            centercrop=True,
    ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        return info

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def read_video_from_file(self, video_path):
        """
        try:
            info = self._get_video_info(video_path)
            h, w = info["height"], info["width"]
        except Exception:
            print('ffprobe failed at: {}'.format(video_path))
            return {'video': torch.zeros(1), 'input': video_path,
                    'info': {}}
        """
        info = self._get_video_info(video_path)
        h, w = info["height"], info["width"]
        height, width = self._get_output_dim(h, w)

        try:
            duration = info["duration"]
            fps = self.framerate
            if duration > 0 and duration < 1/fps+0.1:
                fps = 2/max(int(duration), 1)
                print(duration, fps)
        except Exception:
            fps = self.framerate
        
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)
            .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape(
            [-1, height, width, 3])
        video = torch.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)
        return video


class SlowfastVideoReader:
    """
    Pytorch video loader.
    Copied and modified from:
    https://github.com/linjieli222/HERO_Video_Feature_Extractor/blob/main/slowfast/extract_feature/video_loader.py
    """
    def __init__(
            self,
            framerate=1/2,
            clip_len=2,
            size=224,
            centercrop=True,
    ):
        """
        Args:
        """
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.clip_len = clip_len # preprocess clip_len
        self.preprocess = Preprocessing("3d", target_fps=framerate, size=224, clip_len=clip_len, 
                                        padding_mode='tile', min_num_clips=1)

    def __len__(self):
        return len(self.csv)

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        return info

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def read_video_from_file(self, video_path):
        info = self._get_video_info(video_path)
        h, w = info["height"], info["width"]
        height, width = self._get_output_dim(h, w)
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=self.framerate)
            .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape(
            [-1, height, width, 3])
        video = torch.from_numpy(video)
        video = self.preprocess(video, info)
        return video


def clip_iterator(video, batch_size):
    n_chunk = len(video)
    n_iter = int(math.ceil(n_chunk / float(batch_size)))
    for i in range(n_iter):
        min_ind = i * batch_size
        max_ind = (i + 1) * batch_size
        # Transfer the data to the current GPU device.
        if isinstance(video, (list,)):
            inputs = []
            for i in range(len(video)):
                inputs.append(video[i][min_ind:max_ind])
        else:
            inputs = video[min_ind:max_ind]
        yield min_ind, max_ind, inputs


def pack_pathway_output(frames, device):
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
        ).long().to(device),
    )
    frame_list = [slow_pathway, fast_pathway]
    return frame_list