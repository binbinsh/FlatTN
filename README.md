# FlatTN

This repository contains the FlatTN training/evaluation/inference pipeline for the ICASSP 2022 paper: [An End-to-End Chinese Text Normalization Model based on Rule-Guided Flat-Lattice Transformer](https://arxiv.org/abs/2203.16954).

## 1. Current Official Model (2026-02-09)

Best pretrained model files are distributed via Google Drive:

- Public link: [https://drive.google.com/drive/folders/1I-fNYXBwmeyrTWxHJ0X5ySZ-4ugn2Nb2](https://drive.google.com/drive/folders/1I-fNYXBwmeyrTWxHJ0X5ySZ-4ugn2Nb2)
- After download, place files under: `models/FlatTN-20250209/`

Test metrics (paper + current checkpoint):

| Source | Split | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Paper (Table 2, FlatTN) | `test` | 0.9907 | - | - | 0.9708 |
| Current (`best.pt`) | `test` | 0.9911750541 | 0.9744456275 | 0.9706709707 | 0.9725546366 |

Note: the paper reports only `Accuracy` and `F1` in Table 2.

## 2. Project Layout

- `flattn/`: core model/data/lattice/training implementation.
- `scripts/`: train/evaluate/predict CLI entrypoints.
- `dataset/processed/shuffled_BMESO/`: train/dev/test split used by the current released checkpoint.

## 3. Dataset and Split Policy

We release a large-scale **Chinese Text Normalization (TN) Dataset** in corporatioin with Databaker (Beijing) Technology Co., Ltd.

To download the dataset, please visit https://www.data-baker.com/en/#/data/index/TNtts.

(For Chinese version of the download page, please visit https://www.data-baker.com/data/index/TNtts.)

### Paper-aligned split used by current best run

- `dataset/processed/shuffled_BMESO/train.char.bmes` (`23957` sentences)
- `dataset/processed/shuffled_BMESO/dev.char.bmes` (`2998` sentences)
- `dataset/processed/shuffled_BMESO/test.char.bmes` (`2997` sentences)

Use `dataset/processed/shuffled_BMESO/*` for training/evaluation in this repo.

## 4. Environment

```bash
uv pip install -r requirements.txt
```

## 5. Train (Current Kept Configuration)

```bash
uv run python scripts/train_flattn.py \
  --train_file dataset/processed/shuffled_BMESO/train.char.bmes \
  --dev_file dataset/processed/shuffled_BMESO/dev.char.bmes \
  --test_file dataset/processed/shuffled_BMESO/test.char.bmes \
  --lexicon_path lexicon.txt \
  --bert_name hfl/chinese-bert-wwm \
  --output_dir outputs_flattn_reproduce_official_gpu_paperexact \
  --batch_size 10 \
  --num_epochs 20 \
  --lr 6e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --gradient_accumulation_steps 1 \
  --optim sgd \
  --momentum 0.9 \
  --bert_lr_rate 0.05 \
  --fix_bert_epoch 5 \
  --epoch_lr_decay 0.05 \
  --clip_type value \
  --clip_value 5.0 \
  --head 8 --head_dim 20 --layer 1 --ff 3 \
  --max_seq_len 512 \
  --four_pos_fusion ff_two \
  --use_rel_pos 1 \
  --embed_dropout 0.5 \
  --gaz_dropout 0.5 \
  --output_dropout 0.3 \
  --pre_dropout 0.5 \
  --post_dropout 0.3 \
  --ff_dropout 0.15 \
  --attn_dropout 0.0 \
  --use_lexicon 1 \
  --use_rule 1 \
  --device cuda \
  --seed 42 \
  --init uniform
```

Or run:

```bash
bash train.sh
```

## 6. Evaluate

```bash
uv run python scripts/evaluate_flattn.py \
  --checkpoint models/FlatTN-20250209/best.pt \
  --run_config models/FlatTN-20250209/run_config.json \
  --label_vocab models/FlatTN-20250209/label_vocab.json \
  --test_file dataset/processed/shuffled_BMESO/test.char.bmes \
  --device auto
```

## 7. Released Model

### Model location

- Download source: [https://drive.google.com/drive/folders/1I-fNYXBwmeyrTWxHJ0X5ySZ-4ugn2Nb2](https://drive.google.com/drive/folders/1I-fNYXBwmeyrTWxHJ0X5ySZ-4ugn2Nb2)
- After download, place files under: `models/FlatTN-20250209/`

Model selection:

- `best.pt` is the dev-best checkpoint and should be used for evaluation/inference.
- `last.pt` is only the final-epoch checkpoint for resume-training scenarios.

### Integrity check

```bash
MODEL_DIR=models/FlatTN-20250209
cd "$MODEL_DIR"
shasum -a 256 -c SHA256SUMS.txt
```

## 8. Easy Inference on Raw Text

Use the new inference script:

```bash
uv run python scripts/predict_flattn.py \
  --checkpoint models/FlatTN-20250209/best.pt \
  --run_config models/FlatTN-20250209/run_config.json \
  --label_vocab models/FlatTN-20250209/label_vocab.json \
  --input_file demo_input.txt \
  --output_file demo_output.jsonl \
  --device auto
```

You can also pass one sentence directly:

```bash
uv run python scripts/predict_flattn.py \
  --checkpoint models/FlatTN-20250209/best.pt \
  --run_config models/FlatTN-20250209/run_config.json \
  --label_vocab models/FlatTN-20250209/label_vocab.json \
  --text "今天是2021/09/06。" \
  --device auto
```

## 9. Input/Output Formats

### Training/Eval BMESO format (`*.char.bmes`)

One char + one tag per line, blank line between sentences:

```text
今 O
天 O
是 O
2 B-DIGIT
0 M-DIGIT
2 M-DIGIT
1 E-DIGIT

```

### Inference input text format

`--input_file` should contain one sentence per line:

```text
今天是2021/09/06。
价格是3.14元。
```

### Inference output JSONL format

One JSON object per input sentence:

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Input line index (0-based). |
| `text` | `string` | Original input sentence. |
| `chars` | `string[]` | Character-level tokens. |
| `tags` | `string[]` | Predicted tag for each char (`len(tags) == len(chars)`). |
| `entities` | `object[]` | Decoded entities from tags. |

```json
{
  "id": 0,
  "text": "今天是2021/09/06。",
  "chars": ["今", "天", "是", "2", "0", "2", "1", "/", "0", "9", "/", "0", "6", "。"],
  "tags": ["O", "O", "O", "B-DIGIT", "M-DIGIT", "M-DIGIT", "E-DIGIT", "S-SLASH_OR", "B-DIGIT", "E-DIGIT", "S-SLASH_OR", "B-DIGIT", "E-DIGIT", "S-PUNC"],
  "entities": [
    {"start": 3, "end": 7, "label": "DIGIT", "text": "2021"},
    {"start": 7, "end": 8, "label": "SLASH_OR", "text": "/"},
    {"start": 8, "end": 10, "label": "DIGIT", "text": "09"},
    {"start": 10, "end": 11, "label": "SLASH_OR", "text": "/"},
    {"start": 11, "end": 13, "label": "DIGIT", "text": "06"},
    {"start": 13, "end": 14, "label": "PUNC", "text": "。"}
  ]
}
```

`entities[*].end` is exclusive.
