from __future__ import annotations

"""Utilities for loading and tokenising WikiText-2 for character-level models."""

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List

import numpy as np
from datasets import load_dataset

UNK_CHAR = "?"


@dataclass(frozen=True)
class CharVocab:
    """Character vocabulary with mapping helpers."""

    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        chars = sorted(set(text) | {UNK_CHAR})
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        return cls(stoi=stoi, itos=list(chars))

    @classmethod
    def from_serializable(cls, payload: Dict[str, List[str]]) -> CharVocab:
        itos = payload["itos"]
        stoi = {ch: idx for idx, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=list(itos))

    def to_serializable(self) -> Dict[str, List[str]]:
        return {"itos": list(self.itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: Iterable[str]) -> np.ndarray:
        get_index = self.stoi.get
        unk = self.stoi[UNK_CHAR]
        return np.fromiter((get_index(ch, unk) for ch in text), dtype=np.int64)

    def decode(self, ids: np.ndarray) -> str:
        return "".join(self.itos[int(i)] for i in ids.tolist())


@dataclass
class CharData:
    vocab: CharVocab
    train_tokens: np.ndarray
    val_tokens: np.ndarray


def load_wikitext2_text(split: str) -> str:
    """Download WikiText-2 and concatenate the chosen split into a single string."""

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n".join(example["text"] for example in dataset if example["text"])


def prepare_char_data() -> CharData:
    train_text = load_wikitext2_text("train")
    val_text = load_wikitext2_text("validation")

    vocab = CharVocab.from_text(train_text)
    train_tokens = vocab.encode(train_text)
    val_tokens = vocab.encode(val_text)

    return CharData(vocab=vocab, train_tokens=train_tokens, val_tokens=val_tokens)


def batch_iterator(tokens: np.ndarray, seq_len: int, batch_size: int, *, rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield random batches of input/target sequences for language modelling."""

    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        raise ValueError("Sequence length is too long for the provided token array")

    while True:
        starts = rng.integers(0, max_start, size=batch_size)
        batch_x = np.stack([tokens[s : s + seq_len] for s in starts], axis=0)
        batch_y = np.stack([tokens[s + 1 : s + seq_len + 1] for s in starts], axis=0)
        yield batch_x, batch_y
