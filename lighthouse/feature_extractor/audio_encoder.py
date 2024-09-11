import torch
import librosa
import numpy as np

from lighthouse.feature_extractor.base_encoder import BaseEncoder
from lighthouse.feature_extractor.audio_encoders.pann import Cnn14

from typing import Optional

SAMPLE_RATE: int = 32000
WINDOW_SIZE: int = 1024
HOP_SIZE: int = 320
MEL_BINS: int = 64
FMIN: int = 50
FMAX: int = 14000
CLASSES_NUM: int = 527


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

        self._model = Cnn14(sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, 
                            hop_size=HOP_SIZE, mel_bins=MEL_BINS,
                            fmin=FMIN, fmax=FMAX, classes_num=CLASSES_NUM)
        checkpoint = torch.load(pann_path, map_location=device)
        self._model.load_state_dict(checkpoint['model'])
        self._model.eval()
    
    def _move_data_to_device(
        self,
        x: np.ndarray) -> torch.Tensor:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)
        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)
        else:
            return x
        return x.to(self._device)

    def _select_audio_encoders(self):
        audio_encoders = {
            'clip_slowfast_pann': []
        }

    @torch.no_grad()
    def encode(
        self,
        video_path: str,
        feature_time: int = 2,
        sampling_rate: int = SAMPLE_RATE,
        ) -> torch.Tensor:
        (audio, _) = librosa.core.load(video_path, sr=sampling_rate, mono=True)
        time = audio.shape[-1] / sampling_rate
        batches = int(time // feature_time)
        clip_sr = round(sampling_rate * feature_time)
        assert clip_sr >= 9920, 'clip_sr = round(sampling_rate * feature_time) should be larger than 9920.'
        audio = audio[:batches * clip_sr]
        audio_clips = np.reshape(audio, [batches, clip_sr])
        audio_clips = self._move_data_to_device(audio_clips)
        output_dict = self._model(audio_clips, None)
        return output_dict['embedding'].unsqueeze(0)