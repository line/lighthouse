from typing import Tuple

import torch
from msclap import CLAP

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


class CLAPText:
    def __init__(
        self,
        device: str,
        model_path: str,
    ) -> None:
        self._model_path: str = model_path
        self._device: str = device
        use_cuda = True if self._device == 'cuda' else False
        self._clap_extractor = CLAP(use_cuda=use_cuda, version=model_path)
        self._preprocessor = self._clap_extractor.preprocess_text
        self._text_encoder = self._clap_extractor.clap.caption_encoder

    def __call__(self, query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        preprocessed = self._preprocessor([query])
        mask = preprocessed['attention_mask']

        out = self._text_encoder.base(**preprocessed)
        x = out[0]  # out[1] is pooled output

        return x, mask
