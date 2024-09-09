import abc
import torch

class BaseEncoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(
        self,
        input: str) -> torch.Tensor:
        """
        Input nature is different between text encoder and vision/audio encoders.
        The former expects a query, the latter expects the path to the video and audio file.
        """
        return torch.Tensor([])