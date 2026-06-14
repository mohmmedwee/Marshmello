"""
Causal (masked) self-attention for GPT-style decoder-only models.

Uses PyTorch scaled_dot_product_attention with is_causal=True — the same
pattern as nanoGPT / modern GPT implementations, with better MPS support
than nn.MultiheadAttention + manual -inf masks on Apple Silicon.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_bool_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular mask (True = allowed). Useful for teaching."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Additive causal mask for nn.MultiheadAttention (CPU/CUDA fallback).

    Values: 0 on/below diagonal, large negative above diagonal.
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    if device.type == "mps":
        mask = torch.where(torch.isinf(mask), torch.full_like(mask, -1e4), mask)
    return mask


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (GPT-style, single QKV projection)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        neg = -1e4 if x.device.type == "mps" else float("-inf")
        attn_scores = attn_scores.masked_fill(causal, neg)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(context)
