"""
Phase 10: BPE-level language model

Same task as phases 07 and 08 (predict the next token), but tokens are
BPE *subwords* from phase 09 — the same granularity real LLMs use.

Why this phase exists:
  Phase 07 (char):  readable output is terrible (letter loops)
  Phase 08 (word):  readable but <UNK> on new words, huge word vocab
  Phase 10 (BPE):   readable + handles "ChatGPT" via subword pieces

Pipeline:
  corpus → train BPE → BPE token IDs → tiny transformer → generate subwords → decode
"""

from __future__ import annotations

import copy
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Import shared pieces from earlier phases ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "08_word_level_language_model"))
sys.path.insert(0, str(ROOT / "09_bpe_tokenizer_demo"))

from corpus import build_corpus  # noqa: E402
from bpe_demo import BPETokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset: random slices of BPE token IDs (same idea as phase 08)
# ---------------------------------------------------------------------------
class BPEDataset:
    def __init__(self, token_ids: list[int], vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def get_batch(self, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - block_size - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + block_size] for s in starts])
        y = torch.stack([self.data[s + 1 : s + block_size + 1] for s in starts])
        return x, y


def split_ids(ids: list[int], val_ratio: float = 0.1) -> tuple[list[int], list[int]]:
    split_at = int(len(ids) * (1.0 - val_ratio))
    return ids[:split_at], ids[split_at:]


# ---------------------------------------------------------------------------
# Tiny transformer (same architecture family as phases 08/07)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
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
        h, _ = self.attn(h, h, h)
        x = x + self.dropout1(h)
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


class BPELM(nn.Module):
    """Predict the next BPE subword token."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        block_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_emb = PositionalEncoding(d_model, block_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(idx)
        x = self.emb_dropout(x)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def estimate_loss(
    model: BPELM,
    train_data: BPEDataset,
    val_data: BPEDataset,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_batches: int = 15,
) -> tuple[float, float]:
    model.eval()
    losses: dict[str, float] = {}
    for name, dataset in [("train", train_data), ("val", val_data)]:
        batch_losses = []
        for _ in range(eval_batches):
            x, y = dataset.get_batch(batch_size, block_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))
            batch_losses.append(loss.item())
        losses[name] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses["train"], losses["val"]


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor, recent_ids: list[int], penalty: float = 1.5
) -> torch.Tensor:
    adjusted = logits.clone()
    for token_id, count in Counter(recent_ids).items():
        factor = penalty**count
        if adjusted[token_id] > 0:
            adjusted[token_id] /= factor
        else:
            adjusted[token_id] *= factor
    return adjusted


@torch.no_grad()
def generate(
    model: BPELM,
    bpe: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.9,
    top_k: int = 30,
    repetition_penalty: float = 1.4,
    repetition_window: int = 16,
    seed: int | None = None,
) -> str:
    """
    Autoregressive sampling on BPE token IDs, then decode to text.

    New words like "ChatGPT" work because the prompt is BPE-encoded into
    subwords that exist in vocab — no <UNK> token needed.
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    ids = bpe.encode_ids(prompt)
    if not ids:
        ids = [0]

    generated: list[int] = []

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.block_size :]], dtype=torch.long)
        logits = model(context)[0, -1, :].clone()

        recent = generated[-repetition_window:]
        if recent:
            logits = apply_repetition_penalty(logits, recent, penalty=repetition_penalty)

        logits = logits / temperature
        logits = apply_top_k(logits, top_k)

        if not torch.isfinite(logits).any():
            break

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
        generated.append(next_id)

        # Stop after a word boundary marker when we have enough text
        if bpe.itos[next_id].endswith(bpe.END) and len(generated) >= 12:
            break

    return bpe.decode_ids(ids)


def print_comparison_table() -> None:
    """
    Summarize typical generation quality across phases (from running 07/08/10).
    Phase 10 combines readability with open vocabulary.
    """
    print("\n--- Generation quality: phase 07 vs 08 vs 10 ---")
    print()
    print("| Phase | Token unit   | Typical failure mode              | New word 'ChatGPT' |")
    print("|-------|--------------|-----------------------------------|--------------------|")
    print("| 07    | character    | letter loops: beeeee, rrrrrr        | spelled char-by-char (long, messy) |")
    print("| 08    | whole word   | word loops: to be to be           | <UNK> if unseen    |")
    print("| 10    | BPE subword  | phrase fragments, but readable    | split: C ha t G PT |")
    print()
    print("BPE (phase 10) is closest to what GPT-style models use in production.")


def main() -> None:
    print("Phase 10: BPE-level language model")
    print("=" * 60)

    # --- Step 1: corpus + BPE tokenizer (same as phase 09) ---
    raw_corpus = build_corpus(min_words=10_000)
    num_merges = 800

    print("Training BPE tokenizer on phase 08 corpus...")
    bpe = BPETokenizer()
    bpe.train(raw_corpus, num_merges=num_merges, verbose=False)
    bpe.build_index()

    # --- Step 2: corpus → integer token sequence ---
    all_ids = bpe.corpus_to_ids(raw_corpus)
    train_ids, val_ids = split_ids(all_ids, val_ratio=0.1)

    train_set = BPEDataset(train_ids, bpe.vocab_size)
    val_set = BPEDataset(val_ids, bpe.vocab_size)

    # CPU-friendly settings (vocab ~864, sequences ~15k tokens)
    block_size = 64
    d_model = 128
    num_heads = 4
    num_layers = 2
    d_ff = 256
    batch_size = 32
    learning_rate = 2e-3
    dropout = 0.2
    max_epochs = 200
    eval_every = 25

    device = torch.device("cpu")

    print(f"\nBPE vocab size:   {bpe.vocab_size:,}")
    print(f"Total BPE tokens: {len(all_ids):,}")
    print(f"Train tokens:     {len(train_ids):,}")
    print(f"Val tokens:       {len(val_ids):,}")
    print(f"Block size:       {block_size} subwords")
    print(f"Device:           {device}")

    model = BPELM(
        vocab_size=bpe.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    model.train()
    for epoch in range(1, max_epochs + 1):
        x, y = train_set.get_batch(batch_size, block_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, bpe.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0 or epoch == 1:
            train_loss, val_loss = estimate_loss(
                model, train_set, val_set, block_size, batch_size, device
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1

            print(
                f"Epoch {epoch:4d} | train loss = {train_loss:.4f} | "
                f"val loss = {val_loss:.4f}"
            )

            if stale >= 4:
                print(f"Early stop at epoch {epoch}: val loss plateaued.")
                break

    print(f"\nBest val loss: {best_val:.4f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best checkpoint for generation.")

    # --- Step 3: generate from prompts (including unseen-word prompt) ---
    prompts = ["To be", "All the world's", "ChatGPT tokenization"]
    temperature = 0.85

    print(f"\n--- BPE-level generation (temperature={temperature}, top_k=30) ---")
    for prompt in prompts:
        # Show how the prompt is tokenized (educational)
        pieces = bpe.encode(prompt)
        text = generate(
            model,
            bpe,
            prompt,
            max_new_tokens=55,
            temperature=temperature,
            top_k=30,
            seed=42,
        )
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  BPE in:  {pieces[:14]}{'...' if len(pieces) > 14 else ''}")
        print(f"  Output:  \"{text}\"")

    print_comparison_table()

    print("=" * 60)
    print("Phase 10 complete: same transformer idea as 07/08, better tokens.")


if __name__ == "__main__":
    main()
