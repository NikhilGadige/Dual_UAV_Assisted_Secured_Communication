"""
DeepSC: Deep Learning Enabled Semantic Communication System.

PyTorch re-implementation of the architecture proposed in Xie et al.,
"Deep Learning Enabled Semantic Communication Systems" (IEEE TSP, 2021):

    Semantic Source -> Semantic Encoder (Transformer) -> Channel Encoder (dense)
    -> Physical Channel (AWGN) -> Channel Decoder (dense) -> Semantic Decoder
    (Transformer) -> Recovered Meaning

This module contains only the neural network architecture and a thin
vocabulary helper. Training and evaluation (used to build the
SINR -> semantic-similarity lookup table consumed by the Week 9 ISAC system
model) live in train_deepsc.py, so that the rest of the system model does
not need a torch dependency at simulation time.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


class Vocabulary:
    """Whitespace-tokenized vocabulary built from a list of sentences."""

    def __init__(self, sentences: List[str]):
        tokens = set()
        for sentence in sentences:
            tokens.update(sentence.lower().split())
        vocab_list = SPECIAL_TOKENS + sorted(tokens)
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab_list)}
        self.id_to_token: Dict[int, str] = {i: tok for tok, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.start_id = self.token_to_id[START_TOKEN]
        self.end_id = self.token_to_id[END_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, sentence: str, max_len: int) -> List[int]:
        """Tokenizes a sentence and appends <end>, padded/truncated to max_len."""
        ids = [self.token_to_id.get(w, self.unk_id) for w in sentence.lower().split()]
        ids = ids[: max_len - 1] + [self.end_id]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == self.end_id or i == self.pad_id:
                break
            if i == self.start_id:
                continue
            words.append(self.id_to_token.get(int(i), UNK_TOKEN))
        return " ".join(words)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def power_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalizes channel-encoder output to unit average power per symbol."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return math.sqrt(x.size(-1)) * x / norm


def awgn_channel(x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """
    Additive White Gaussian Noise channel.
    snr_db: scalar or per-batch tensor broadcastable to x's batch dimension.
    Assumes unit average transmit power per symbol (post power_normalize).
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_std = torch.sqrt(1.0 / snr_linear).clamp_min(1e-6)
    while noise_std.dim() < x.dim():
        noise_std = noise_std.unsqueeze(-1)
    noise = torch.randn_like(x) * noise_std
    return x + noise


class DeepSC(nn.Module):
    """
    DeepSC semantic communication model: Transformer-based semantic
    encoder/decoder wrapped around a dense channel encoder/decoder and an
    AWGN physical channel.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        channel_dim: int = 16,
        max_len: int = 16,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.semantic_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, channel_dim),
        )
        self.channel_decoder = nn.Sequential(
            nn.Linear(channel_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.semantic_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode_to_channel_symbols(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Semantic encoder -> channel encoder -> power-normalized symbols."""
        src_pad_mask = src_ids.eq(self.pad_id)
        x = self.pos_encoding(self.embedding(src_ids))
        enc_out = self.semantic_encoder(x, src_key_padding_mask=src_pad_mask)
        symbols = power_normalize(self.channel_encoder(enc_out))
        return symbols, src_pad_mask

    def decode_from_channel(
        self, channel_out: torch.Tensor, src_pad_mask: torch.Tensor, tgt_in_ids: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.channel_decoder(channel_out)
        tgt_pad_mask = tgt_in_ids.eq(self.pad_id)
        tgt_len = tgt_in_ids.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt_in_ids.device), diagonal=1).bool()

        tgt = self.pos_encoding(self.embedding(tgt_in_ids))
        dec_out = self.semantic_decoder(
            tgt, memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(dec_out)

    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        symbols, src_pad_mask = self.encode_to_channel_symbols(src_ids)
        received = awgn_channel(symbols, snr_db)
        return self.decode_from_channel(received, src_pad_mask, tgt_in_ids)

    @torch.no_grad()
    def sentence_embedding(self, src_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pooled semantic-encoder embedding (no channel), used as the
        semantic-similarity proxy in place of an external BERT model."""
        src_pad_mask = src_ids.eq(self.pad_id)
        x = self.pos_encoding(self.embedding(src_ids))
        enc_out = self.semantic_encoder(x, src_key_padding_mask=src_pad_mask)
        mask = (~src_pad_mask).unsqueeze(-1).float()
        summed = (enc_out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    @torch.no_grad()
    def greedy_decode(
        self, src_ids: torch.Tensor, snr_db: float, vocab: Vocabulary, max_len: int,
    ) -> torch.Tensor:
        """Runs the full pipeline (encoder -> channel -> decoder) and
        autoregressively decodes the reconstructed sentence."""
        device = src_ids.device
        symbols, src_pad_mask = self.encode_to_channel_symbols(src_ids)
        snr_tensor = torch.full((src_ids.size(0),), float(snr_db), device=device)
        received = awgn_channel(symbols, snr_tensor)
        memory = self.channel_decoder(received)

        batch = src_ids.size(0)
        tgt_ids = torch.full((batch, 1), vocab.start_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_pad_mask = tgt_ids.eq(self.pad_id)
            causal_mask = torch.triu(
                torch.ones(tgt_ids.size(1), tgt_ids.size(1), device=device), diagonal=1
            ).bool()
            tgt = self.pos_encoding(self.embedding(tgt_ids))
            dec_out = self.semantic_decoder(
                tgt, memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask,
            )
            logits = self.output_proj(dec_out[:, -1:, :])
            next_id = logits.argmax(dim=-1)
            tgt_ids = torch.cat([tgt_ids, next_id], dim=1)
        return tgt_ids[:, 1:]
