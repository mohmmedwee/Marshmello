"""
Phase 06: One transformer block (PyTorch)

A standard encoder block contains:

  1. Token embedding + positional encoding
  2. Multi-head self-attention (here: single head for simplicity)
  3. Residual connection + LayerNorm
  4. Feed-forward network (FFN)
  5. Another residual + LayerNorm

We run a forward pass on a tiny sentence and print tensor shapes
so you can see data flow through the block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Adds position information to token embeddings.

    Transformers have no built-in word order — attention is permutation-
    invariant without this. We add a fixed sine/cosine pattern per position.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape (1, max_len, d_model) for batch broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class FeedForward(nn.Module):
    """
    Position-wise FFN: two linear layers with ReLU in the middle.

    Applied independently to each token vector.
    Typically expands dimension (d_model → 4*d_model) then projects back.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One encoder-style block:

      x → LayerNorm → Self-Attn → +x (residual)
        → LayerNorm → FFN       → +x (residual)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # input shape: (batch, seq, feature)
        )
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Sub-layer 1: self-attention with pre-norm residual ---
        # Pre-norm: normalize BEFORE the sub-layer (common in modern models)
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed, normed, normed, need_weights=True)
        attn_out = self.dropout(attn_out)
        x = x + attn_out  # residual: keep original signal + add attention update

        # --- Sub-layer 2: feed-forward with pre-norm residual ---
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        ffn_out = self.dropout(ffn_out)
        x = x + ffn_out

        return x, attn_weights


class MiniTransformer(nn.Module):
    """
    Minimal stack: embedding → positional encoding → one transformer block.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.block = TransformerBlock(d_model, num_heads, d_ff)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # token_ids: (batch, seq_len)
        x = self.token_embedding(token_ids)   # → (batch, seq, d_model)
        x = self.pos_encoding(x)
        x, attn_weights = self.block(x)
        return x, attn_weights


def build_vocab(sentence: str) -> tuple[dict[str, int], list[str]]:
    """Simple word-level vocab for demo."""
    words = sentence.split()
    word_to_id = {w: i for i, w in enumerate(sorted(set(words)))}
    return word_to_id, words


def main() -> None:
    print("Phase 06: Mini transformer block (PyTorch)")
    print("=" * 60)

    sentence = "I love AI"
    word_to_id, _ = build_vocab(sentence)
    id_to_word = {v: k for k, v in word_to_id.items()}
    token_ids = torch.tensor([[word_to_id[w] for w in sentence.split()]])

    d_model = 32
    num_heads = 4
    d_ff = 64
    vocab_size = len(word_to_id)

    model = MiniTransformer(vocab_size, d_model, num_heads, d_ff)
    model.eval()

    print(f"\nSentence: \"{sentence}\"")
    print(f"Token IDs: {token_ids.tolist()}")
    print(f"Vocab: {word_to_id}")
    print(f"d_model={d_model}, heads={num_heads}, d_ff={d_ff}\n")

    with torch.no_grad():
        output, attn_weights = model(token_ids)

    print("--- Shape trace ---")
    print(f"  Input token_ids:     {tuple(token_ids.shape)}")
    print(f"  After embedding:     (batch, seq, d_model) = (1, 3, {d_model})")
    print(f"  Block output:        {tuple(output.shape)}")
    print(f"  Attention weights:   {tuple(attn_weights.shape)}  (batch, seq, seq)")

    print("\n--- Output vectors (first 6 dims of each token) ---")
    for i, tid in enumerate(token_ids[0].tolist()):
        word = id_to_word[tid]
        preview = output[0, i, :6].numpy()
        print(f"  {word:8s}: {preview}")

    print("\n--- Attention weights (query × key), row = query token ---")
    # PyTorch may return (batch, seq, seq) or (batch, heads, seq, seq)
    if attn_weights.dim() == 4:
        avg_attn = attn_weights[0].mean(dim=0).numpy()
    else:
        avg_attn = attn_weights[0].numpy()
    tokens = sentence.split()
    header = "         " + "  ".join(f"{t:>8s}" for t in tokens)
    print(header)
    for i, t in enumerate(tokens):
        row = "  ".join(f"{avg_attn[i, j]:8.3f}" for j in range(len(tokens)))
        print(f"  {t:8s} {row}")

    print("\n" + "=" * 60)
    print("Components used: embedding, positional encoding, self-attention,")
    print("feed-forward, residual connections, layer normalization.")


if __name__ == "__main__":
    main()
