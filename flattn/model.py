from __future__ import annotations

import collections
import inspect
from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from .modules import Transformer_Encoder, get_embedding, seq_len_to_mask

try:
    # Prefer TorchCRF for consistency with local experiments.
    from TorchCRF import CRF as _CRF  # type: ignore
except ImportError:
    from torchcrf import CRF as _CRF  # type: ignore


@dataclass
class FlatTNConfig:
    bert_name: str = "hfl/chinese-bert-wwm"
    head: int = 8
    head_dim: int = 20
    layer: int = 1
    ff: int = 3
    use_rel_pos: bool = True
    max_seq_len: int = 512
    four_pos_fusion: str = "ff_two"
    embed_dropout: float = 0.5
    gaz_dropout: float = 0.5
    output_dropout: float = 0.3
    pre_dropout: float = 0.5
    post_dropout: float = 0.3
    ff_dropout: float = 0.15
    attn_dropout: float = 0.0
    pos_norm: bool = False
    layer_preprocess: str = ""
    layer_postprocess: str = "an"
    scaled: bool = False
    attn_ff: bool = False
    ff_activate: str = "relu"
    k_proj: bool = False
    q_proj: bool = True
    v_proj: bool = True
    r_proj: bool = True
    four_pos_fusion_shared: bool = True
    bert_pool_method: str = "first"
    finetune_bert: bool = True

    @property
    def hidden_size(self) -> int:
        return self.head * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_size"] = self.hidden_size
        return payload


class BERTLatticeEmbedding(nn.Module):
    """Generate token representations by pooling BERT subword outputs per lattice token."""

    def __init__(self, bert_name: str, pool_method: str = "first"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.hidden_size = self.bert.config.hidden_size
        allowed = {"first", "last", "avg", "max"}
        if pool_method not in allowed:
            raise ValueError(f"Unsupported pool method: {pool_method}. Choices: {sorted(allowed)}")
        self.pool_method = pool_method

    def forward(self, batch_tokens: list[list[str]]) -> torch.Tensor:
        if not batch_tokens:
            raise ValueError("batch_tokens is empty")

        enc = self.tokenizer(
            batch_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.bert.config.max_position_embeddings,
        )

        word_ids_batch = [enc.word_ids(batch_index=i) for i in range(len(batch_tokens))]
        device = next(self.bert.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        hidden = self.bert(**enc).last_hidden_state
        pooled_tokens: list[torch.Tensor] = []

        for sample_idx, word_ids in enumerate(word_ids_batch):
            tok_to_pieces: dict[int, list[int]] = {}
            for piece_idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                tok_to_pieces.setdefault(wid, []).append(piece_idx)

            token_reps = []
            for tok_idx in range(len(batch_tokens[sample_idx])):
                piece_ids = tok_to_pieces.get(tok_idx)
                if not piece_ids:
                    token_reps.append(torch.zeros(self.hidden_size, device=device, dtype=hidden.dtype))
                    continue
                piece_tensor = hidden[sample_idx, piece_ids, :]
                if self.pool_method == "first":
                    token_reps.append(piece_tensor[0])
                elif self.pool_method == "last":
                    token_reps.append(piece_tensor[-1])
                elif self.pool_method == "max":
                    token_reps.append(piece_tensor.max(dim=0).values)
                else:
                    token_reps.append(piece_tensor.mean(dim=0))
            pooled_tokens.append(torch.stack(token_reps, dim=0))

        return pad_sequence(pooled_tokens, batch_first=True)


class FlatTNModel(nn.Module):
    def __init__(self, config: FlatTNConfig, label_size: int):
        super().__init__()
        self.config = config
        self.label_size = label_size

        self.bert_embed = BERTLatticeEmbedding(config.bert_name, pool_method=config.bert_pool_method)
        bert_hidden = self.bert_embed.hidden_size

        self.char_proj = nn.Linear(bert_hidden, config.hidden_size)
        self.lex_proj = nn.Linear(bert_hidden, config.hidden_size)

        dropout_dict = {
            "embed": config.embed_dropout,
            "gaz": config.gaz_dropout,
            "output": config.output_dropout,
            "pre": config.pre_dropout,
            "post": config.post_dropout,
            "ff": config.ff_dropout,
            "attn": config.attn_dropout,
        }

        pe = pe_ss = pe_se = pe_es = pe_ee = None
        if config.use_rel_pos:
            pe = get_embedding(config.max_seq_len, config.hidden_size, rel_pos_init=1)
            if config.pos_norm:
                pe_sum = pe.sum(dim=-1, keepdim=True)
                pe = pe / pe_sum
            pe_ss = pe_se = pe_es = pe_ee = pe

        self.encoder = Transformer_Encoder(
            config.hidden_size,
            config.head,
            config.layer,
            relative_position=config.use_rel_pos,
            learnable_position=False,
            add_position=False,
            layer_preprocess_sequence=config.layer_preprocess,
            layer_postprocess_sequence=config.layer_postprocess,
            dropout=dropout_dict,
            scaled=config.scaled,
            ff_size=config.hidden_size * config.ff,
            mode=collections.defaultdict(bool),
            dvc="cpu",
            max_seq_len=config.max_seq_len,
            pe=pe,
            pe_ss=pe_ss,
            pe_se=pe_se,
            pe_es=pe_es,
            pe_ee=pe_ee,
            k_proj=config.k_proj,
            q_proj=config.q_proj,
            v_proj=config.v_proj,
            r_proj=config.r_proj,
            attn_ff=config.attn_ff,
            ff_activate=config.ff_activate,
            lattice=True,
            four_pos_fusion=config.four_pos_fusion,
            four_pos_fusion_shared=config.four_pos_fusion_shared,
        )
        self._pe_tensors = {
            "pe": pe,
            "pe_ss": pe_ss,
            "pe_se": pe_se,
            "pe_es": pe_es,
            "pe_ee": pe_ee,
        }

        self.output_dropout = nn.Dropout(config.output_dropout)
        self.output = nn.Linear(config.hidden_size, label_size)
        crf_init_params = inspect.signature(_CRF.__init__).parameters
        if "batch_first" in crf_init_params:
            self.crf = _CRF(label_size, batch_first=True)
        elif "use_gpu" in crf_init_params:
            self.crf = _CRF(label_size, use_gpu=False)
        else:
            self.crf = _CRF(label_size)

        self._crf_forward_has_reduction = "reduction" in inspect.signature(self.crf.forward).parameters
        self._crf_has_decode = hasattr(self.crf, "decode")
        self._crf_has_viterbi_decode = hasattr(self.crf, "viterbi_decode")

    def _sync_relative_tensors(self, device: torch.device) -> None:
        if self._pe_tensors["pe"] is not None:
            self.encoder.pe = self._pe_tensors["pe"].to(device)
            self.encoder.pe_ss = self._pe_tensors["pe_ss"].to(device)
            self.encoder.pe_se = self._pe_tensors["pe_se"].to(device)
            self.encoder.pe_es = self._pe_tensors["pe_es"].to(device)
            self.encoder.pe_ee = self._pe_tensors["pe_ee"].to(device)

        if hasattr(self.encoder, "four_pos_fusion_embedding"):
            for attr in ["pe", "pe_ss", "pe_se", "pe_es", "pe_ee"]:
                tensor = getattr(self.encoder.four_pos_fusion_embedding, attr, None)
                if tensor is not None:
                    setattr(self.encoder.four_pos_fusion_embedding, attr, tensor.to(device))

    def forward(
        self,
        lattice_tokens: list[list[str]],
        pos_s: torch.Tensor,
        pos_e: torch.Tensor,
        seq_len: torch.Tensor,
        lex_num: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device

        batch_embed = self.bert_embed(lattice_tokens)
        if not self.config.finetune_bert:
            batch_embed = batch_embed.detach()
        batch_embed = batch_embed.to(device)

        pos_s = pos_s.to(device)
        pos_e = pos_e.to(device)
        seq_len = seq_len.to(device)
        lex_num = lex_num.to(device)
        labels = labels.to(device) if labels is not None else None

        self._sync_relative_tensors(device)

        max_lattice = batch_embed.size(1)
        char_mask = seq_len_to_mask(seq_len, max_len=max_lattice).bool()
        total_mask = seq_len_to_mask(seq_len + lex_num, max_len=max_lattice).bool()
        lex_mask = total_mask & (~char_mask)

        embed_char = self.char_proj(batch_embed)
        embed_lex = self.lex_proj(batch_embed)

        embed_char = embed_char.masked_fill(~char_mask.unsqueeze(-1), 0.0)
        embed_lex = embed_lex.masked_fill(~lex_mask.unsqueeze(-1), 0.0)
        embedding = embed_char + embed_lex

        encoded = self.encoder(embedding, seq_len, lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)

        max_char = int(seq_len.max().item())
        encoded = encoded[:, :max_char, :]
        emissions = self.output(self.output_dropout(encoded))
        mask = seq_len_to_mask(seq_len, max_len=max_char).bool()

        if labels is not None:
            labels = labels[:, :max_char]
            safe_labels = labels.masked_fill(labels < 0, 0)
            if self._crf_forward_has_reduction:
                loss = -self.crf(emissions, safe_labels, mask=mask, reduction="mean")
            else:
                # Some legacy CRF implementations return per-sample log-likelihood.
                raw = self.crf(emissions, safe_labels, mask)
                loss = -raw.mean() if raw.dim() > 0 else -raw
            return {"loss": loss, "emissions": emissions, "mask": mask}

        if self._crf_has_decode:
            pred = self.crf.decode(emissions, mask=mask)
        elif self._crf_has_viterbi_decode:
            pred = self.crf.viterbi_decode(emissions, mask)
        else:
            raise RuntimeError("CRF backend provides neither decode nor viterbi_decode.")
        return {"pred": pred, "emissions": emissions, "mask": mask}
