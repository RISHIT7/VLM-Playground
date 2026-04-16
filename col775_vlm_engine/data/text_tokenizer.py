import torch
import json
from typing import Tuple, List

class CLEVRTokenizer:
    def __init__(self, max_seq_len: int = 77):
        self.max_seq_len = max_seq_len
        self.word2idx = {"[PAD]": 0, "[UNK]": 1, "[SOS]": 2}
        self.idx2word = {0: "[PAD]", 1: "[UNK]", 2: "[SOS]"}
        self.vocab_size = 3  # will be updated after build_vocab

    def build_vocab(self, annotations_file: str):
        """Build vocabulary from all captions in the training JSON."""
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        for item in data:
            caption = item.get("caption", "")
            for word in caption.lower().split():
                word = word.strip(".,!?")
                if word not in self.word2idx and word not in ["[PAD]", "[UNK]", "[SOS]"]:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.idx2word)] = word
        self.word2idx["[EOS]"] = len(self.word2idx)
        self.idx2word[len(self.idx2word)] = "[EOS]"
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Returns:
            tokens: (max_seq_len,) LongTensor, padded with 0.
            padding_mask: (max_seq_len,) BoolTensor, True for pad positions.
        """
        words = ["[SOS]"] + [w.strip(".,!?") for w in text.lower().split()] + ["[EOS]"]
        ids = [self.word2idx.get(w, self.word2idx["[UNK]"]) for w in words]
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
            ids[-1] = self.word2idx["[EOS]"]
        tokens = torch.zeros(self.max_seq_len, dtype=torch.long)
        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        for i, idx in enumerate(ids):
            tokens[i] = idx
            padding_mask[i] = False
        return tokens, padding_mask

    def decode(self, tokens: torch.LongTensor) -> str:
        words = []
        for idx in tokens:
            if idx.item() == self.word2idx["[PAD]"]:
                break
            if idx.item() in self.idx2word:
                words.append(self.idx2word[idx.item()])
        return " ".join(words)