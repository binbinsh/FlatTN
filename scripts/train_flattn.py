#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattn.data import FlatTNDataset, build_label_vocab, flattn_collate_fn, read_bmes
from flattn.lattice import build_trie, load_lexicon_words
from flattn.model import FlatTNConfig, FlatTNModel
from flattn.training import pick_device, train_model
from flattn.rules import Rule


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model_like_paper(model: torch.nn.Module, init_style: str) -> None:
    style = init_style.lower()
    if style == "none":
        return

    with torch.no_grad():
        initialized = 0
        for name, param in model.named_parameters():
            lname = name.lower()
            if (
                "bert" in lname
                or "embedding" in lname
                or "pos" in lname
                or "pe" in lname
                or "bias" in lname
                or "crf" in lname
                or param.dim() <= 1
            ):
                continue
            if style == "uniform":
                torch.nn.init.xavier_uniform_(param)
            elif style == "norm":
                torch.nn.init.xavier_normal_(param)
            else:
                raise ValueError(f"Unsupported --init: {init_style}")
            initialized += 1
    print(f"Initialized {initialized} parameter tensors with xavier_{style}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FlatTN (paper-aligned FLAT + rule-guided lattice)")

    parser.add_argument("--train_file", default="dataset/processed/shuffled_BMESO/train.char.bmes")
    parser.add_argument("--dev_file", default="dataset/processed/shuffled_BMESO/dev.char.bmes")
    parser.add_argument("--test_file", default="dataset/processed/shuffled_BMESO/test.char.bmes")
    parser.add_argument("--lexicon_path", default="lexicon.txt")
    parser.add_argument("--output_dir", default="outputs_flattn_reimpl")

    parser.add_argument("--bert_name", default="hfl/chinese-bert-wwm")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--bert_lr_rate", type=float, default=0.05)
    parser.add_argument("--fix_bert_epoch", type=int, default=5)
    parser.add_argument("--epoch_lr_decay", type=float, default=0.05)
    parser.add_argument("--clip_type", default="value", choices=["norm", "value"])
    parser.add_argument("--clip_value", type=float, default=5.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=20)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--ff", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--four_pos_fusion", default="ff_two")
    parser.add_argument("--use_rel_pos", type=int, default=1)
    parser.add_argument("--pos_norm", type=int, default=0)
    parser.add_argument("--layer_preprocess", default="")
    parser.add_argument("--layer_postprocess", default="an")
    parser.add_argument("--ff_activate", default="relu")
    parser.add_argument("--k_proj", type=int, default=0)
    parser.add_argument("--q_proj", type=int, default=1)
    parser.add_argument("--v_proj", type=int, default=1)
    parser.add_argument("--r_proj", type=int, default=1)
    parser.add_argument("--bert_pool_method", default="first", choices=["first", "last", "avg", "max"])

    parser.add_argument("--embed_dropout", type=float, default=0.5)
    parser.add_argument("--gaz_dropout", type=float, default=0.5)
    parser.add_argument("--output_dropout", type=float, default=0.3)
    parser.add_argument("--pre_dropout", type=float, default=0.5)
    parser.add_argument("--post_dropout", type=float, default=0.3)
    parser.add_argument("--ff_dropout", type=float, default=0.15)
    parser.add_argument("--attn_dropout", type=float, default=0.0)

    parser.add_argument("--use_lexicon", type=int, default=1)
    parser.add_argument("--use_rule", type=int, default=1)
    parser.add_argument("--max_lattice_len", type=int, default=0)
    parser.add_argument("--finetune_bert", type=int, default=1)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--eval_test_each_epoch", type=int, default=1)
    parser.add_argument("--max_train_sentences", type=int, default=0)
    parser.add_argument("--max_dev_sentences", type=int, default=0)
    parser.add_argument("--max_test_sentences", type=int, default=0)
    parser.add_argument("--init", default="uniform", choices=["uniform", "norm", "none"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    full_train_sentences = read_bmes(args.train_file)
    full_dev_sentences = read_bmes(args.dev_file)
    full_test_sentences = read_bmes(args.test_file)

    # Keep label space fixed to the whole dataset even when training with a subset.
    label2id, id2label = build_label_vocab(full_train_sentences + full_dev_sentences + full_test_sentences)

    train_sentences = full_train_sentences
    dev_sentences = full_dev_sentences
    test_sentences = full_test_sentences

    if args.max_train_sentences > 0:
        train_sentences = train_sentences[: args.max_train_sentences]
    if args.max_dev_sentences > 0:
        dev_sentences = dev_sentences[: args.max_dev_sentences]
    if args.max_test_sentences > 0:
        test_sentences = test_sentences[: args.max_test_sentences]

    lexicon_words = load_lexicon_words(args.lexicon_path, min_len=2)
    trie = build_trie(lexicon_words)
    rule = Rule()

    max_lattice_len = args.max_lattice_len if args.max_lattice_len > 0 else args.max_seq_len

    train_dataset = FlatTNDataset(
        sentences=train_sentences,
        label2id=label2id,
        trie=trie,
        rule=rule,
        use_lexicon=bool(args.use_lexicon),
        use_rule=bool(args.use_rule),
        max_lattice_len=max_lattice_len,
    )
    dev_dataset = FlatTNDataset(
        sentences=dev_sentences,
        label2id=label2id,
        trie=trie,
        rule=rule,
        use_lexicon=bool(args.use_lexicon),
        use_rule=bool(args.use_rule),
        max_lattice_len=max_lattice_len,
    )
    test_dataset = FlatTNDataset(
        sentences=test_sentences,
        label2id=label2id,
        trie=trie,
        rule=rule,
        use_lexicon=bool(args.use_lexicon),
        use_rule=bool(args.use_rule),
        max_lattice_len=max_lattice_len,
    )

    print("Train summary:", train_dataset.summary)
    print("Dev summary:", dev_dataset.summary)
    print("Test summary:", test_dataset.summary)

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": flattn_collate_fn,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    model_config = FlatTNConfig(
        bert_name=args.bert_name,
        head=args.head,
        head_dim=args.head_dim,
        layer=args.layer,
        ff=args.ff,
        use_rel_pos=bool(args.use_rel_pos),
        max_seq_len=args.max_seq_len,
        four_pos_fusion=args.four_pos_fusion,
        embed_dropout=args.embed_dropout,
        gaz_dropout=args.gaz_dropout,
        output_dropout=args.output_dropout,
        pre_dropout=args.pre_dropout,
        post_dropout=args.post_dropout,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        pos_norm=bool(args.pos_norm),
        layer_preprocess=args.layer_preprocess,
        layer_postprocess=args.layer_postprocess,
        ff_activate=args.ff_activate,
        k_proj=bool(args.k_proj),
        q_proj=bool(args.q_proj),
        v_proj=bool(args.v_proj),
        r_proj=bool(args.r_proj),
        bert_pool_method=args.bert_pool_method,
        finetune_bert=bool(args.finetune_bert),
    )

    model = FlatTNModel(config=model_config, label_size=len(label2id))
    init_model_like_paper(model, args.init)

    report = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        id2label=id2label,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        clip_type=args.clip_type,
        clip_value=args.clip_value,
        optimizer_name=args.optim,
        momentum=args.momentum,
        bert_lr_rate=args.bert_lr_rate,
        fix_bert_epoch=args.fix_bert_epoch,
        epoch_lr_decay=args.epoch_lr_decay,
        device=device,
        log_steps=args.log_steps,
        eval_test_each_epoch=bool(args.eval_test_each_epoch),
    )

    label_json = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with (output_dir / "label_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(label_json, f, ensure_ascii=False, indent=2)

    run_config = {
        "args": vars(args),
        "model_config": model_config.to_dict(),
        "device": str(device),
        "num_labels": len(label2id),
        "train_summary": train_dataset.summary.__dict__,
        "dev_summary": dev_dataset.summary.__dict__,
        "test_summary": test_dataset.summary.__dict__,
        "best_dev_f1": report["best_dev_f1"],
        "best_epoch": report["best_epoch"],
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("Training completed.")
    print(json.dumps({
        "best_epoch": report["best_epoch"],
        "best_dev_f1": report["best_dev_f1"],
        "best_test_metrics": report["best_test_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
