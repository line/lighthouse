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

import torch
from tqdm import tqdm


class PostProcessorDETR:
    def __init__(self, clip_length=2, min_ts_val=0, max_ts_val=150,
                 min_w_l=2, max_w_l=70, move_window_method="center",
                 process_func_names=("clip_window_l", "clip_ts", "round_multiple")):
        self.clip_length = clip_length
        self.min_ts_val = min_ts_val
        self.max_ts_val = max_ts_val
        self.min_w_l = min_w_l
        self.max_w_l = max_w_l
        self.move_window_method = move_window_method
        self.process_func_names = process_func_names
        self.name2func = dict(
            clip_ts=self.clip_min_max_timestamps,
            round_multiple=self.round_to_multiple_clip_lengths,
        )

    def __call__(self, lines):
        processed_lines = []
        for line in tqdm(lines, desc=f"convert to multiples of clip_length={self.clip_length}"):
            windows_and_scores = torch.tensor(line["pred_relevant_windows"])
            windows = windows_and_scores[:, :2]
            for func_name in self.process_func_names:
                windows = self.name2func[func_name](windows)
            line["pred_relevant_windows"] = torch.cat(
                [windows, windows_and_scores[:, 2:3]], dim=1).tolist()
            line["pred_relevant_windows"] = [e[:2] + [float(f"{e[2]:.4f}")] for e in line["pred_relevant_windows"]]
            processed_lines.append(line)
        return processed_lines

    def clip_min_max_timestamps(self, windows):
        """
        windows: (#windows, 2)  torch.Tensor
        ensure timestamps for all windows is within [min_val, max_val], clip is out of boundaries.
        """
        return torch.clamp(windows, min=self.min_ts_val, max=self.max_ts_val)

    def round_to_multiple_clip_lengths(self, windows):
        """
        windows: (#windows, 2)  torch.Tensor
        ensure the final window timestamps are multiples of `clip_length`
        """
        return torch.round(windows / self.clip_length) * self.clip_length

