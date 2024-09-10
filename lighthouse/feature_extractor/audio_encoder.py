import torch

from lighthouse.feature_extractor.base_encoder import BaseEncoder

class AudioEncoder(BaseEncoder):
    def __init__(self) -> None:
        pass

    def encode(self) -> torch.Tensor:
        return torch.Tensor([])