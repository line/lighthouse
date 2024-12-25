from typing import Optional

import numpy as np
import torch
import torchaudio.transforms as T
from msclap import CLAP


class CLAPAudioConfig:
    def __init__(self, cfg: Optional[dict] = None):
        self.sample_rate: int = 44100
        self.window_sec: float = 1.0
        self.version: str = '2023'
        self.feature_time: float = 1.0

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class CLAPAudio(torch.nn.Module):
    def __init__(self, device: str, cfg: CLAPAudioConfig):
        super(CLAPAudio, self).__init__()
        use_cuda = True if device == 'cuda' else False
        self.clap = CLAP(use_cuda=use_cuda, version=cfg.version)
        self.sample_rate = cfg.sample_rate
        self.window_sec = cfg.window_sec
        self.feature_time = cfg.feature_time
        self._device = device

    def _preprocess(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        audio_tensor = self._move_data_to_device(audio)
        resampler = T.Resample(sr, self.sample_rate).to(self._device)
        audio_tensor = resampler(audio_tensor)  # original implementation in msclap

        win_length = int(round(self.window_sec * self.sample_rate))
        hop_length = int(round(self.feature_time * self.sample_rate))

        time = audio_tensor.shape[-1] / self.sample_rate
        batches = int(time // self.feature_time)
        clip_sr = round(self.sample_rate * self.feature_time)
        audio_tensor = audio_tensor[:batches * clip_sr] # Truncate audio to fit the clip_sr

        audio_clip = audio_tensor.unfold(0, win_length, hop_length)

        return audio_clip

    def _move_data_to_device(
        self,
        x: np.ndarray) -> torch.Tensor:
        if 'float' in str(x.dtype):
            return torch.Tensor(x).to(self._device)
        elif 'int' in str(x.dtype):
            return torch.LongTensor(x).to(self._device)
        else:
            raise ValueError('The input x cannot be cast into float or int.')

    def forward(self, audio: np.ndarray, sr: int):
        audio_clip = self._preprocess(audio, sr)
        output_dict = self.clap.clap.audio_encoder.base(audio_clip)
        audio_mask = torch.ones(1, len(output_dict['embedding'])).to(self._device)
        x = output_dict['embedding'].unsqueeze(0)
        return x, audio_mask
