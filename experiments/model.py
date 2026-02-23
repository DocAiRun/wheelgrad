"""
experiments/model.py
─────────────────────
Shared mini-transformer definition used by both training scripts.
Parameterized to swap Standard vs Wheel layers easily.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerBlock(nn.Module):
    """
    One transformer block. Arithmetic mode selectable at init.
    
    mode='standard' : uses nn.Softmax + nn.LayerNorm (may produce NaN)
    mode='wheel'    : uses WheelSoftmax + WheelLayerNorm (algebraically stable)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        mode: Literal['standard', 'wheel'] = 'standard'
    ):
        super().__init__()
        self.mode = mode
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV + output projections (same for both modes)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

        # The key difference: softmax + norm
        if mode == 'standard':
            self.softmax = nn.Softmax(dim=-1)
            self.norm1   = nn.LayerNorm(d_model)
            self.norm2   = nn.LayerNorm(d_model)
        else:
            from wheelgrad.torch_ops import WheelSoftmax, WheelLayerNorm
            self.softmax = WheelSoftmax(dim=-1)
            self.norm1   = WheelLayerNorm(d_model)
            self.norm2   = WheelLayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Multi-head self-attention
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = reshape(self.q(x))
        K = reshape(self.k(x))
        V = reshape(self.v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores + mask  # additive causal mask

        weights = self.softmax(scores.view(-1, T, T)).view(B, self.n_heads, T, T)
        attn = torch.matmul(self.drop(weights), V)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        attn = self.out(attn)

        # Residual + norm 1
        x = self.norm1(x + attn)

        # Feed-forward + residual + norm 2
        ff = self.ff2(self.act(self.ff1(x)))
        x = self.norm2(x + ff)

        return x


class MiniTransformer(nn.Module):
    """
    Small transformer for stability experiments.
    
    Vocabulary classification task:
        Input  : (B, T) token indices
        Output : (B, T, vocab_size) logits
    
    Parameters
    ----------
    mode : 'standard' or 'wheel'
        Which arithmetic to use in attention + normalization.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.0,
        mode: Literal['standard', 'wheel'] = 'standard'
    ):
        super().__init__()
        self.mode = mode
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = PositionalEncoding(d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, mode)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.pos(self.embed(tokens))
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Upper triangular causal mask (-inf above diagonal)."""
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device),
        diagonal=1
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def make_dataset(
    n_samples: int = 512,
    seq_len: int = 32,
    vocab_size: int = 256,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic next-token prediction dataset.
    Task: predict next token given previous tokens.
    """
    torch.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    x = tokens[:, :-1]  # (N, T)   inputs
    y = tokens[:, 1:]   # (N, T)   targets (next token)
    return x, y
