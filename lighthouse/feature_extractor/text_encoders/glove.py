import torch
from typing import Tuple
from torchtext import vocab

class GloVe:
    def __init__(
        self,
        device: str,
        model_path: str,
        ) -> None:
        self._model_path: str = model_path
        self._device: str = device
        self._vocab = self._initialize_vocab()
        self._embedding = self._initialize_embedding()

    def _initialize_vocab(self):
        _vocab = vocab.pretrained_aliases[self._model_path]()
        _vocab.itos.extend(['<unk>'])
        _vocab.stoi['<unk>'] = _vocab.vectors.shape[0]
        return _vocab
    
    def _initialize_embedding(self):
        self._vocab.vectors = torch.cat(
            (self.vocab.vectors, torch.zeros(1, self._vocab.dim)), dim=0)
        embedding = torch.nn.Embedding.from_pretrained(self._vocab.vectors)
        return embedding

    def __call__(
        self,
        query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        WATCH: query should be tokens with spaces.
        e.g., 'a man is speaking in front of the camera'
        """
        word_inds = torch.LongTensor(
            [self._vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        mask = torch.ones((1, word_inds.shape[0])).to(self._device)
        return self._embedding(word_inds).unsqueeze(0).to(self._device), mask