import torch
import librosa

import numpy as np

from lighthouse.panns.models import Cnn14

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x
    return x.to(device)

class PannExtractor:
    """
    PANNs audio feature extractor
    """
    def __init__(self, pann_path, device):
        self.pann_path = pann_path
        self.device = device
        self.model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        checkpoint = torch.load(pann_path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    @torch.no_grad()
    def extract_feature(self, video_path, feature_time=2, sr=32000):
        (audio, _) = librosa.core.load(video_path, sr=32000, mono=True)
        time = audio.shape[-1] / sr
        batches = int(time // feature_time)
        clip_sr = round(sr * feature_time)
        assert clip_sr >= 9920
        audio = audio[:batches*clip_sr]
        audio_clips = np.reshape(audio, [batches, clip_sr])
        audio_clips = move_data_to_device(audio_clips, device=self.device)
        output_dict = self.model(audio_clips, None)
        return output_dict['embedding'].unsqueeze(0)