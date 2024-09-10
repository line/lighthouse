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
import torch
import ffmpeg
import math
import numpy as np

from typing import Optional, Dict, Union, Tuple

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

class CLIPLoader:
    def __init__(
        self,
        framerate: float,
        size: int,
        device: str,
        centercrop: bool = True) -> None:
        self._framerate = framerate
        self._size = size
        self._device = device
        self._centercrop = centercrop
    
    def _video_info(
        self,
        video_path: str) -> Optional[Dict[str, Union[int, float]]]:
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
            return None
        
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        
        return info

    def _output_dim(
        self,
        h: int,
        w: int) -> Tuple[int, int]:
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)        
    
    def __call__(
        self,
        video_path: str) -> Optional[torch.Tensor]:
        
        info = self._video_info(video_path)
        if info is None:
            return None

        h, w = info["height"], info["width"]
        height, width = self._output_dim(h, w)

        try:
            duration = info["duration"]
            fps = self._framerate
            
            if duration > 0 and duration < 1 / fps + 0.1:
                fps = 2/max(int(duration), 1)
        
        except Exception:
            return None
        
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)
            .filter('scale', width, height)
        )

        if self.centercrop:
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
        video = torch.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)
        return video