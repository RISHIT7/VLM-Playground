import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

class CLEVRTokenizer:
    def __init__(self, max_seq_len: int = 77):
        self.max_seq_len = max_seq_len
        self.word2idx = {"[PAD]": 0, "[UNK]": 1, "[SOS]": 2}
        self.idx2word = {0: "[PAD]", 1: "[UNK]", 2: "[SOS]"}
        self.vocab_size = 3  # will be updated after build_vocab

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "max_seq_len": self.max_seq_len,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "vocab_size": self.vocab_size,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.max_seq_len = int(state.get("max_seq_len", self.max_seq_len))
        if "word2idx" in state:
            self.word2idx = {str(k): int(v) for k, v in state["word2idx"].items()}
        if "idx2word" in state:
            self.idx2word = {int(k): v for k, v in state["idx2word"].items()}
        self.vocab_size = int(state.get("vocab_size", len(self.word2idx)))

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "CLEVRTokenizer":
        tokenizer = cls(max_seq_len=int(state.get("max_seq_len", 77)))
        tokenizer.load_state_dict(state)
        return tokenizer

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(self.to_state_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CLEVRTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return cls.from_state_dict(state)

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

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for idx in tokens.tolist():
            if idx == self.word2idx["[PAD]"]:
                break
            if idx in self.idx2word:
                words.append(self.idx2word[idx])
        return " ".join(words)