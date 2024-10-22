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

This code is from: https://github.com/qiuqiangkong/panns_inference

This models.py contains selected models from: 
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class PANNConfig:
    def __init__(self, cfg: Optional[dict] = None):
        self.sample_rate: int = 32000
        self.window_size: int = 1024
        self.hop_size: int = 320
        self.mel_bins: int = 64
        self.fmin: int = 50
        self.fmax: int = 14000
        self.classes_num: int = 527
        self.model_path: Optional[str] = None
        self.feature_time: float = 2.0

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class PANN(torch.nn.Module):
    def __init__(self, device: str, cfg: PANNConfig):
        super(PANN, self).__init__()
        self._device: str = device
        self.sample_rate: int = cfg.sample_rate
        self.feature_time: float = cfg.feature_time
        self._model = Cnn14(
            sample_rate=cfg.sample_rate,
            window_size=cfg.window_size,
            hop_size=cfg.hop_size,
            mel_bins=cfg.mel_bins,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            classes_num=cfg.classes_num,
        )

        if cfg.model_path is not None:
            checkpoint = torch.load(cfg.model_path, map_location=device)
            self._model.load_state_dict(checkpoint['model'])
            self._model.eval()
            self._model.to(device)
        else:
            raise TypeError('pann_path should not be None when using AudioEncoder.')

    def _preprocess(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        time = audio.shape[-1] / self.sample_rate
        batches = int(time // self.feature_time)
        clip_sr = round(self.sample_rate * self.feature_time)
        assert clip_sr >= 9920, 'clip_sr = round(sampling_rate * feature_time) should be larger than 9920.'
        audio = audio[:batches * clip_sr] # Truncate audio to fit the clip_sr
        audio_clip = audio.reshape([batches, clip_sr])

        audio_clip_tensor = self._move_data_to_device(audio_clip)
        return audio_clip_tensor

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
        output_dict = self._model(audio_clip, None) # audio_clip: (batch_size, clip_samples)
        audio_mask = torch.ones(1, len(output_dict['embedding'])).to(self._device)
        x = output_dict['embedding'].unsqueeze(0)
        return x, audio_mask



def do_mixup(x, mixup_lambda):
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + \
        x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    return out.transpose(0, -1)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
