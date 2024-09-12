import torch
import librosa
import numpy as np

from lighthouse.feature_extractor.base_encoder import BaseEncoder
from lighthouse.feature_extractor.audio_encoders.pann import Cnn14

from typing import Optional, Tuple


class AudioEncoder(BaseEncoder):
    SAMPLE_RATE: int = 32000
    WINDOW_SIZE: int = 1024
    HOP_SIZE: int = 320
    MEL_BINS: int = 64
    FMIN: int = 50
    FMAX: int = 14000
    CLASSES_NUM: int = 527

    def __init__(
        self,
        feature_name: str,
        device: str,
        pann_path: Optional[str]) -> None:
        self._feature_name = feature_name
        self._device = device
        self._pann_path = pann_path

        self._model = Cnn14(sample_rate=self.SAMPLE_RATE, window_size=self.WINDOW_SIZE,
                            hop_size=self.HOP_SIZE, mel_bins=self.MEL_BINS,
                            fmin=self.FMIN, fmax=self.FMAX, classes_num=self.CLASSES_NUM)
        
        if pann_path is not None:
            checkpoint = torch.load(pann_path, map_location=device)
            self._model.load_state_dict(checkpoint['model'])
            self._model.eval()
        else:
            raise TypeError('pann_path should not be None when using AudioEncoder.')
    
    def _move_data_to_device(
        self,
        x: np.ndarray) -> torch.Tensor:
        if 'float' in str(x.dtype):
            return torch.Tensor(x).to(self._device)
        elif 'int' in str(x.dtype):
            return torch.LongTensor(x).to(self._device)
        else:
            raise ValueError('The input x cannot be cast into float or int.')

    @torch.no_grad()
    def encode(
        self,
        video_path: str,
        feature_time: int = 2,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        (audio, _) = librosa.core.load(video_path, sr=self.SAMPLE_RATE, mono=True)
        time = audio.shape[-1] / self.SAMPLE_RATE
        batches = int(time // feature_time)
        clip_sr = round(self.SAMPLE_RATE * feature_time)
        assert clip_sr >= 9920, 'clip_sr = round(sampling_rate * feature_time) should be larger than 9920.'
        audio = audio[:batches * clip_sr]
        audio_clips = np.reshape(audio, [batches, clip_sr])
        audio_clip_tensor = self._move_data_to_device(audio_clips)
        output_dict = self._model(audio_clip_tensor, None)
        audio_mask = torch.ones(1, len(output_dict['embedding'])).to(self._device)
        return output_dict['embedding'].unsqueeze(0), audio_mask