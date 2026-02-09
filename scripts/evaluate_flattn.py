#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattn.data import FlatTNDataset, flattn_collate_fn, read_bmes
from flattn.lattice import build_trie, load_lexicon_words
from flattn.model import FlatTNConfig, FlatTNModel
from flattn.training import evaluate_model, pick_device
from flattn.rules import Rule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained FlatTN checkpoint")
    parser.add_argument("--checkpoint", default="models/FlatTN-20250209/best.pt")
    parser.add_argument("--run_config", default="models/FlatTN-20250209/run_config.json")
    parser.add_argument("--label_vocab", default="models/FlatTN-20250209/label_vocab.json")

    parser.add_argument("--test_file", default="dataset/processed/shuffled_BMESO/test.char.bmes")
    parser.add_argument("--lexicon_path", default="lexicon.txt")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max_lattice_len", type=int, default=0)

    return parser.parse_args()


def _load_model_config(run_config_path: Path) -> FlatTNConfig:
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    cfg = payload.get("model_config", {})
    cfg.pop("hidden_size", None)
    return FlatTNConfig(**cfg)


def _load_vocab(label_vocab_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(label_vocab_path.read_text(encoding="utf-8"))
    label2id = {str(k): int(v) for k, v in payload["label2id"].items()}
    id2label = {int(k): str(v) for k, v in payload["id2label"].items()}
    return label2id, id2label


def _adapt_crf_state_dict(state_dict: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Bridge CRF parameter naming differences across backends:
    - TorchCRF: start_trans / end_trans / trans_matrix
    - torchcrf: start_transitions / end_transitions / transitions
    """
    model_keys = set(model.state_dict().keys())
    has_legacy = {"crf.start_trans", "crf.end_trans", "crf.trans_matrix"} <= set(state_dict.keys())
    has_standard = {"crf.start_transitions", "crf.end_transitions", "crf.transitions"} <= set(state_dict.keys())
    expects_legacy = "crf.start_trans" in model_keys
    expects_standard = "crf.start_transitions" in model_keys

    if has_legacy and expects_standard:
        mapped = dict(state_dict)
        mapped["crf.start_transitions"] = mapped.pop("crf.start_trans")
        mapped["crf.end_transitions"] = mapped.pop("crf.end_trans")
        mapped["crf.transitions"] = mapped.pop("crf.trans_matrix")
        return mapped

    if has_standard and expects_legacy:
        mapped = dict(state_dict)
        mapped["crf.start_trans"] = mapped.pop("crf.start_transitions")
        mapped["crf.end_trans"] = mapped.pop("crf.end_transitions")
        mapped["crf.trans_matrix"] = mapped.pop("crf.transitions")
        return mapped

    return state_dict


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    run_config_path = Path(args.run_config).expanduser()
    label_vocab_path = Path(args.label_vocab).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()

    model_config = _load_model_config(run_config_path)
    label2id, id2label = _load_vocab(label_vocab_path)

    lexicon_words = load_lexicon_words(Path(args.lexicon_path).expanduser(), min_len=2)
    trie = build_trie(lexicon_words)
    rule = Rule()

    max_lattice_len = args.max_lattice_len if args.max_lattice_len > 0 else model_config.max_seq_len

    test_sentences = read_bmes(Path(args.test_file).expanduser())
    test_dataset = FlatTNDataset(
        sentences=test_sentences,
        label2id=label2id,
        trie=trie,
        rule=rule,
        use_lexicon=True,
        use_rule=True,
        max_lattice_len=max_lattice_len,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=flattn_collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = FlatTNModel(config=model_config, label_size=len(label2id))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = _adapt_crf_state_dict(checkpoint["model_state_dict"], model)
    model.load_state_dict(model_state, strict=True)
    model.to(device)

    metrics = evaluate_model(model, test_loader, id2label=id2label, device=device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
