import torch
import ffmpeg
import numpy as np
from typing import Optional
from lighthouse.frame_loaders.base_loader import BaseLoader

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

class CLIPLoader(BaseLoader):
    def __init__(
        self,
        clip_len: float,
        framerate: float,
        size: int,
        device: str,
        centercrop: bool = True) -> None:
        super().__init__(clip_len, framerate, size, device, centercrop)
        assert self._clip_len == 1. / self._framerate, 'clip_len and inverse of framerate should be equal for CLIPLoader.'

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
        video_tensor = torch.from_numpy(video.astype('float32'))
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        return video_tensor