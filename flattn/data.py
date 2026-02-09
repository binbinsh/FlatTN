from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .lattice import build_lattice


def read_bmes(path: str | Path) -> list[list[tuple[str, str]]]:
    sentences: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line.strip():
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            cur.append((parts[0], parts[1]))
    if cur:
        sentences.append(cur)
    return sentences


def build_label_vocab(sentences: list[list[tuple[str, str]]]) -> tuple[dict[str, int], dict[int, str]]:
    labels = []
    for sent in sentences:
        labels.extend(tag for _, tag in sent)
    uniq = sorted(dict.fromkeys(labels))
    label2id = {label: idx for idx, label in enumerate(uniq)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


@dataclass
class DatasetSummary:
    num_sentences: int
    num_chars: int
    max_char_len: int
    max_lattice_len: int
    avg_lattice_len: float


def summarize_dataset(samples: list[dict]) -> DatasetSummary:
    if not samples:
        return DatasetSummary(0, 0, 0, 0, 0.0)
    num_chars = sum(len(x["chars"]) for x in samples)
    max_char_len = max(len(x["chars"]) for x in samples)
    lattice_lens = [len(x["lattice_tokens"]) for x in samples]
    return DatasetSummary(
        num_sentences=len(samples),
        num_chars=num_chars,
        max_char_len=max_char_len,
        max_lattice_len=max(lattice_lens),
        avg_lattice_len=round(sum(lattice_lens) / len(lattice_lens), 4),
    )


class FlatTNDataset(Dataset):
    def __init__(
        self,
        sentences: list[list[tuple[str, str]]],
        label2id: dict[str, int],
        trie,
        rule,
        use_lexicon: bool = True,
        use_rule: bool = True,
        max_lattice_len: int | None = None,
    ):
        self.samples: list[dict] = []
        self.unknown_label_counter: Counter[str] = Counter()

        for sent in sentences:
            chars = [c for c, _ in sent]
            tags = [t for _, t in sent]
            lattice = build_lattice(
                chars=chars,
                trie=trie,
                rule=rule,
                use_lexicon=use_lexicon,
                use_rule=use_rule,
                max_lattice_len=max_lattice_len,
            )

            label_ids = []
            for tag in tags:
                if tag not in label2id:
                    self.unknown_label_counter[tag] += 1
                    label_ids.append(label2id.get("O", 0))
                else:
                    label_ids.append(label2id[tag])

            self.samples.append(
                {
                    "chars": chars,
                    "labels": label_ids,
                    "lattice_tokens": lattice.tokens,
                    "pos_s": lattice.pos_s,
                    "pos_e": lattice.pos_e,
                    "seq_len": len(chars),
                    "lex_num": lattice.lex_num,
                }
            )

        self.summary = summarize_dataset(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def flattn_collate_fn(batch: list[dict], pad_label_id: int = -100) -> dict:
    max_lattice = max(len(x["lattice_tokens"]) for x in batch)
    max_chars = max(len(x["labels"]) for x in batch)

    pos_s = []
    pos_e = []
    labels = []

    for sample in batch:
        cur_len = len(sample["lattice_tokens"])
        pad_tok = max_lattice - cur_len
        pos_s.append(sample["pos_s"] + [0] * pad_tok)
        pos_e.append(sample["pos_e"] + [0] * pad_tok)

        char_len = len(sample["labels"])
        labels.append(sample["labels"] + [pad_label_id] * (max_chars - char_len))

    return {
        "lattice_tokens": [x["lattice_tokens"] for x in batch],
        "pos_s": torch.tensor(pos_s, dtype=torch.long),
        "pos_e": torch.tensor(pos_e, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "seq_len": torch.tensor([x["seq_len"] for x in batch], dtype=torch.long),
        "lex_num": torch.tensor([x["lex_num"] for x in batch], dtype=torch.long),
    }
