from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def pick_device(preferred: str = "auto") -> torch.device:
    pref = preferred.lower()
    if pref == "cuda":
        return torch.device("cuda")
    if pref == "mps":
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ["pos_s", "pos_e", "labels", "seq_len", "lex_num"]:
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device)
    return out


def _seqeval_tag(tag: str) -> str:
    if tag.startswith("M-"):
        return "I-" + tag[2:]
    return tag


def evaluate_model(model, dataloader: DataLoader, id2label: dict[int, str], device: torch.device) -> dict[str, Any]:
    model.eval()

    gold_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
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

            labels = batch["labels"].detach().cpu()
            seq_lens = batch["seq_len"].detach().cpu().tolist()

            for i, pred_seq in enumerate(predictions):
                seq_len = int(seq_lens[i])
                gold_ids = labels[i, :seq_len].tolist()
                gold_seq = [id2label[idx] for idx in gold_ids]
                pred_seq_tags = [id2label[idx] for idx in pred_seq[:seq_len]]

                gold_tags.append([_seqeval_tag(x) for x in gold_seq])
                pred_tags.append([_seqeval_tag(x) for x in pred_seq_tags])

                for g, p in zip(gold_seq, pred_seq_tags):
                    total += 1
                    if g == p:
                        correct += 1

    precision = precision_score(gold_tags, pred_tags)
    recall = recall_score(gold_tags, pred_tags)
    f1 = f1_score(gold_tags, pred_tags)
    accuracy = correct / max(total, 1)

    report = classification_report(gold_tags, pred_tags, output_dict=True, zero_division=0)
    per_category_f1 = {}
    for key, value in report.items():
        if key in {"micro avg", "macro avg", "weighted avg"}:
            continue
        if isinstance(value, dict) and "f1-score" in value:
            per_category_f1[key] = value["f1-score"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_category_f1": per_category_f1,
        "num_tokens": total,
        "num_sentences": len(gold_tags),
    }


def _set_bert_trainable(model, trainable: bool) -> None:
    if not hasattr(model, "bert_embed") or not hasattr(model.bert_embed, "bert"):
        return
    for p in model.bert_embed.bert.parameters():
        p.requires_grad = trainable


def _build_optimizer(
    model,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    bert_lr_rate: float,
):
    bert_params: list[torch.nn.Parameter] = []
    if hasattr(model, "bert_embed") and hasattr(model.bert_embed, "bert"):
        bert_params = list(model.bert_embed.bert.parameters())
    bert_param_ids = {id(p) for p in bert_params}
    non_bert_params = [p for p in model.parameters() if id(p) not in bert_param_ids]

    param_groups = [{"params": non_bert_params}]
    if bert_params:
        param_groups.append({"params": bert_params, "lr": learning_rate * bert_lr_rate})

    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(
    optimizer,
    updates_per_epoch: int,
    num_epochs: int,
    warmup_ratio: float,
    epoch_lr_decay: float,
):
    total_updates = max(updates_per_epoch * num_epochs, 1)
    warmup_steps = int(total_updates * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            warmup_scale = float(step + 1) / float(max(warmup_steps, 1))
        else:
            warmup_scale = 1.0

        epoch_progress = step / float(max(updates_per_epoch, 1))
        decay_scale = 1.0 / (1.0 + max(epoch_lr_decay, 0.0) * epoch_progress)
        return warmup_scale * decay_scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_model(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    id2label: dict[int, str],
    output_dir: str | Path,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    clip_type: str,
    clip_value: float,
    optimizer_name: str,
    momentum: float,
    bert_lr_rate: float,
    fix_bert_epoch: int,
    epoch_lr_decay: float,
    device: torch.device,
    log_steps: int = 100,
    eval_test_each_epoch: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    if fix_bert_epoch > 0:
        _set_bert_trainable(model, False)

    optimizer = _build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        bert_lr_rate=bert_lr_rate,
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(gradient_accumulation_steps, 1))
    scheduler = _build_scheduler(
        optimizer,
        updates_per_epoch=updates_per_epoch,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        epoch_lr_decay=epoch_lr_decay,
    )

    history: list[dict[str, Any]] = []
    best_dev_f1 = -1.0
    best_epoch = -1
    best_dev_metrics: dict[str, Any] | None = None
    best_test_metrics: dict[str, Any] | None = None

    global_step = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        if fix_bert_epoch > 0 and epoch == fix_bert_epoch + 1:
            _set_bert_trainable(model, True)
            print(f"Unfroze BERT parameters at epoch {epoch}.")

        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            batch = _batch_to_device(batch, device)
            output = model(
                lattice_tokens=batch["lattice_tokens"],
                pos_s=batch["pos_s"],
                pos_e=batch["pos_e"],
                seq_len=batch["seq_len"],
                lex_num=batch["lex_num"],
                labels=batch["labels"],
            )

            loss = output["loss"] / max(gradient_accumulation_steps, 1)
            loss.backward()
            running_loss += float(loss.item())

            should_step = step % gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                if clip_type == "value":
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if should_step and global_step > 0 and global_step % max(log_steps, 1) == 0:
                avg_loss = running_loss / max(step, 1)
                progress.set_postfix(loss=avg_loss)
                # Emit a plain log line as well so redirected logs show steady progress.
                lr_values = ",".join(f"{pg['lr']:.8f}" for pg in optimizer.param_groups)
                print(
                    f"[Epoch {epoch} Step {step}/{len(train_loader)}] "
                    f"global_step={global_step} loss={avg_loss:.6f} lrs=[{lr_values}]"
                )

        train_loss = running_loss / max(len(train_loader), 1)
        dev_metrics = evaluate_model(model, dev_loader, id2label=id2label, device=device)
        test_metrics = None
        if eval_test_each_epoch:
            test_metrics = evaluate_model(model, test_loader, id2label=id2label, device=device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev": dev_metrics,
            "test": test_metrics,
            "global_step": global_step,
        }
        history.append(epoch_record)

        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_dev_f1": best_dev_f1,
        }
        torch.save(last_ckpt, output_dir / "last.pt")

        if dev_metrics["f1"] > best_dev_f1:
            if test_metrics is None:
                test_metrics = evaluate_model(model, test_loader, id2label=id2label, device=device)
            best_dev_f1 = dev_metrics["f1"]
            best_epoch = epoch
            best_dev_metrics = dev_metrics
            best_test_metrics = test_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_dev_f1": best_dev_f1,
                },
                output_dir / "best.pt",
            )

        if test_metrics is None:
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.6f} "
                f"dev_f1={dev_metrics['f1']:.6f} dev_acc={dev_metrics['accuracy']:.6f}"
            )
        else:
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.6f} "
                f"dev_f1={dev_metrics['f1']:.6f} dev_acc={dev_metrics['accuracy']:.6f} "
                f"test_f1={test_metrics['f1']:.6f} test_acc={test_metrics['accuracy']:.6f}"
            )

    elapsed_sec = time.time() - start_time
    final_report = {
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "best_dev_metrics": best_dev_metrics,
        "best_test_metrics": best_test_metrics,
        "history": history,
        "elapsed_seconds": elapsed_sec,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    return final_report
