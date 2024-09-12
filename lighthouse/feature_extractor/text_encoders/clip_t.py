import torch
import clip
from typing import Tuple

class CLIPText:
    def __init__(
        self,
        device: str,
        model_path: str,
        ) -> None:
        self._model_path: str = model_path
        self._device: str = device
        self._clip_extractor, _ = clip.load(model_path, device=device, jit=False)
        self._tokenizer = clip.tokenize

    @property
    def _dtype(self):
        return self._clip_extractor.visual.conv1.weight.dtype

    def __call__(
        self,
        query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self._tokenizer(query).to(self._device)
        x = self._clip_extractor.token_embedding(text).type(self._dtype)
        x = x + self._clip_extractor.positional_embedding.type(self._dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self._clip_extractor.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self._clip_extractor.ln_final(x).type(torch.float32)
        mask = (text != 0).type(torch.float32).to(self._device)
        return x, mask