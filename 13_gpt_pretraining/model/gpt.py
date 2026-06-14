"""
Decoder-only GPT language model.

Architecture (matches real GPT pretraining, not instruction tuning):

    token IDs
      → token embedding
      → learned positional embedding
      → N × causal transformer blocks
      → final LayerNorm
      → LM head (linear → vocab logits)

Training objective: predict the next token (cross-entropy on all positions).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        block_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide num_heads ({num_heads})")

        self.block_size = block_size
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-style small normal init for stable pretraining."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"seq_len {seq_len} exceeds block_size {self.block_size}")

        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate_step_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits for the next token given context (batch, seq)."""
        return self(idx)[:, -1, :]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
