import argparse
from pathlib import Path

import numpy as np
import librosa
from msclap import CLAP
import torch
from torch.nn import functional as F
from tqdm import tqdm


class CLAPAudioConfig:
    def __init__(self, cfg: dict = None):
        self.sample_rate: int = 32000
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

    def _preprocess(self, audio: np.ndarray, sr: int):
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        audio = self._move_data_to_device(audio)

        win_length = int(round(self.window_sec * self.sample_rate))
        hop_length = int(round(self.feature_time * self.sample_rate))

        time = audio.shape[-1] / self.sample_rate
        batches = int(time // self.feature_time)
        clip_sr = round(self.sample_rate * self.feature_time)
        audio = audio[:batches * clip_sr] 
        audio_clip = audio.unfold(0, win_length, hop_length)

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
