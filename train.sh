#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs_flattn_reproduce_official_gpu_paperexact}"
DEVICE="${DEVICE:-cuda}"

uv run python scripts/train_flattn.py \
  --train_file dataset/processed/shuffled_BMESO/train.char.bmes \
  --dev_file dataset/processed/shuffled_BMESO/dev.char.bmes \
  --test_file dataset/processed/shuffled_BMESO/test.char.bmes \
  --lexicon_path lexicon.txt \
  --bert_name hfl/chinese-bert-wwm \
  --output_dir "${OUT_DIR}" \
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
  --device "${DEVICE}" \
  --seed 42 \
  --init uniform

uv run python scripts/evaluate_flattn.py \
  --checkpoint "${OUT_DIR}/best.pt" \
  --run_config "${OUT_DIR}/run_config.json" \
  --label_vocab "${OUT_DIR}/label_vocab.json" \
  --test_file dataset/processed/shuffled_BMESO/test.char.bmes \
  --device "${DEVICE}"
