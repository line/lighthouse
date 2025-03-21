import torch
import librosa

from lighthouse.feature_extractor.base_encoder import BaseEncoder
from lighthouse.feature_extractor.audio_encoders.pann import PANN, PANNConfig
from lighthouse.feature_extractor.audio_encoders.clap_a import CLAPAudio, CLAPAudioConfig

from typing import Optional, Tuple

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


class AudioEncoder(BaseEncoder):
    def __init__(
        self,
        feature_name: str,
        device: str,
        pann_path: Optional[str]) -> None:
        self._feature_name = feature_name
        self._device = device
        self._pann_path = pann_path
        self._audio_encoders = self._select_audio_encoders()

    def _select_audio_encoders(self):
        audio_encoders = {
            'clip_slowfast_pann': [PANN],
            'clap': [CLAPAudio]
        }

        config_dict = {
            'clip_slowfast_pann': [PANNConfig(dict(model_path=self._pann_path))],
            'clap': [CLAPAudioConfig()],
        }

        audio_encoders = [encoder(self._device, cfg)
                         for encoder, cfg in zip(audio_encoders[self._feature_name], config_dict[self._feature_name])]
        return audio_encoders

    @torch.no_grad()
    def encode(
        self,
        video_path: str,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, sr = librosa.core.load(video_path, sr=None, mono=True)

        outputs = [encoder(audio, sr) for encoder in self._audio_encoders]
        audio_features = torch.cat([o[0] for o in outputs])
        audio_masks = torch.cat([o[1] for o in outputs])
        return audio_features, audio_masks
