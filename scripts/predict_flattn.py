#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattn.data import flattn_collate_fn
from flattn.lattice import build_lattice, build_trie, load_lexicon_words
from flattn.model import FlatTNConfig, FlatTNModel
from flattn.training import pick_device
from flattn.rules import Rule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tags/entities with a trained FlatTN checkpoint")
    parser.add_argument("--checkpoint", default="models/FlatTN-20250209/best.pt")
    parser.add_argument("--run_config", default="models/FlatTN-20250209/run_config.json")
    parser.add_argument("--label_vocab", default="models/FlatTN-20250209/label_vocab.json")

    parser.add_argument("--lexicon_path", default="lexicon.txt")
    parser.add_argument("--input_file", default="", help="Input text file, one sentence per line")
    parser.add_argument("--text", action="append", default=[], help="Single text input (repeatable)")
    parser.add_argument("--output_file", default="", help="Output jsonl file path; stdout if empty")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max_lattice_len", type=int, default=0)

    return parser.parse_args()


def _load_model_config(run_config_path: Path) -> tuple[FlatTNConfig, dict[str, Any]]:
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    cfg = dict(payload.get("model_config", {}))
    cfg.pop("hidden_size", None)
    return FlatTNConfig(**cfg), payload


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


def _load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []

    if args.input_file:
        for line in Path(args.input_file).expanduser().read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(line)

    for item in args.text:
        item = item.strip()
        if item:
            texts.append(item)

    if not texts:
        raise ValueError("No input text found. Use --input_file and/or --text.")

    return texts


def _build_samples(
    texts: list[str],
    trie,
    rule,
    use_lexicon: bool,
    use_rule: bool,
    max_lattice_len: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        chars = list(text)
        lattice = build_lattice(
            chars=chars,
            trie=trie,
            rule=rule,
            use_lexicon=use_lexicon,
            use_rule=use_rule,
            max_lattice_len=max_lattice_len,
        )
        samples.append(
            {
                "sample_id": idx,
                "text": text,
                "chars": chars,
                "labels": [0] * len(chars),
                "lattice_tokens": lattice.tokens,
                "pos_s": lattice.pos_s,
                "pos_e": lattice.pos_e,
                "seq_len": len(chars),
                "lex_num": lattice.lex_num,
            }
        )
    return samples


def _decode_entities(chars: list[str], tags: list[str]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    i = 0
    n = len(tags)

    while i < n:
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            i += 1
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "S":
            entities.append({"start": i, "end": i + 1, "label": label, "text": "".join(chars[i : i + 1])})
            i += 1
            continue

        if prefix == "B":
            j = i + 1
            closed = False
            while j < n:
                next_tag = tags[j]
                if "-" not in next_tag:
                    break
                next_prefix, next_label = next_tag.split("-", 1)
                if next_label != label:
                    break
                if next_prefix in {"M", "I"}:
                    j += 1
                    continue
                if next_prefix == "E":
                    j += 1
                    closed = True
                    break
                break

            if not closed:
                # Robust fallback for malformed BMES sequence.
                j = max(j, i + 1)

            entities.append({"start": i, "end": j, "label": label, "text": "".join(chars[i:j])})
            i = j
            continue

        # Fallback for unexpected prefixes (M/E/I without leading B).
        entities.append({"start": i, "end": i + 1, "label": label, "text": "".join(chars[i : i + 1])})
        i += 1

    return entities


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ["pos_s", "pos_e", "seq_len", "lex_num"]:
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device)
    return out


def main() -> None:
    args = parse_args()

    device = pick_device(args.device)
    run_config_path = Path(args.run_config)
    label_vocab_path = Path(args.label_vocab)
    checkpoint_path = Path(args.checkpoint)
    run_config_path = run_config_path.expanduser()
    label_vocab_path = label_vocab_path.expanduser()
    checkpoint_path = checkpoint_path.expanduser()

    model_config, run_payload = _load_model_config(run_config_path)
    _, id2label = _load_vocab(label_vocab_path)

    run_args = run_payload.get("args", {})
    use_lexicon = bool(int(run_args.get("use_lexicon", 1)))
    use_rule = bool(int(run_args.get("use_rule", 1)))

    max_lattice_len = args.max_lattice_len if args.max_lattice_len > 0 else model_config.max_seq_len

    texts = _load_texts(args)
    lexicon_words = load_lexicon_words(Path(args.lexicon_path).expanduser(), min_len=2)
    trie = build_trie(lexicon_words)
    rule = Rule()

    samples = _build_samples(
        texts=texts,
        trie=trie,
        rule=rule,
        use_lexicon=use_lexicon,
        use_rule=use_rule,
        max_lattice_len=max_lattice_len,
    )

    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=flattn_collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = FlatTNModel(config=model_config, label_size=len(id2label))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = _adapt_crf_state_dict(checkpoint["model_state_dict"], model)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    outputs: list[dict[str, Any]] = []
    cursor = 0

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            output = model(
                lattice_tokens=batch["lattice_tokens"],
                pos_s=batch["pos_s"],
                pos_e=batch["pos_e"],
                seq_len=batch["seq_len"],
                lex_num=batch["lex_num"],
                labels=None,
            )

            predictions = output["pred"]
            seq_lens = batch["seq_len"].detach().cpu().tolist()

            for i, pred_seq in enumerate(predictions):
                sample = samples[cursor]
                seq_len = int(seq_lens[i])
                tags = [id2label[idx] for idx in pred_seq[:seq_len]]
                entities = _decode_entities(sample["chars"], tags)
                outputs.append(
                    {
                        "id": sample["sample_id"],
                        "text": sample["text"],
                        "chars": sample["chars"],
                        "tags": tags,
                        "entities": entities,
                    }
                )
                cursor += 1

    lines = [json.dumps(item, ensure_ascii=False) for item in outputs]
    if args.output_file:
        out_path = Path(args.output_file).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} predictions to: {out_path}")
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
