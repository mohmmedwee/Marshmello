"""Pre-norm transformer block with causal multi-head self-attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """
    One GPT block:

        x → LayerNorm → Causal Self-Attention → + residual
          → LayerNorm → Feed-Forward (GELU)    → + residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x
