"""
Phase 08: Word-level language model

Phase 07 predicted the next CHARACTER. Characters are tiny (a-z, punctuation),
so the model often loops on single letters: "eeee", "rrrr".

Phase 08 predicts the next WORD (or punctuation token). Each step chooses a whole
word like "the" or "fortune", so output looks more like broken prose than keyboard
smash — though small data still causes repeated phrases.

Pipeline:
  text → regex tokenizer → token IDs → transformer → next token logits
"""

import copy
import math
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from corpus import build_corpus


# ---------------------------------------------------------------------------
# Regex tokenizer: split into words, punctuation, and newlines as separate tokens
# ---------------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"\n|\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    Examples:
      "Hello, world!" → ["Hello", ",", "world", "!"]
      "Line one\\nLine two" → ["Line", "one", "\\n", "Line", "two"]
    """
    return TOKEN_PATTERN.findall(text)


class WordTokenizer:
    """Build vocabulary from corpus tokens and encode/decode strings."""

    PAD = "<PAD>"

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {self.PAD: 0}
        self.itos: dict[int, str] = {0: self.PAD}

    def build_vocab(self, tokens: list[str]) -> None:
        for token in tokens:
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        """Join tokens back into readable text with sensible spacing."""
        if not ids:
            return ""

        no_space_before = {",", ".", ";", ":", "!", "?", ")", "]", "}", "'"}
        no_space_after = {"(", "[", "{"}

        parts: list[str] = []
        prev = ""
        for token_id in ids:
            if token_id == 0:
                continue
            token = self.itos[token_id]

            if token == "\n":
                parts.append("\n")
            elif not parts or prev == "\n":
                parts.append(token)
            elif token in no_space_before:
                parts.append(token)
            elif prev in no_space_after:
                parts.append(token)
            else:
                parts.append(" " + token)

            prev = token

        return "".join(parts)


class TokenDataset:
    """Random contiguous slices of token IDs for language-model training."""

    def __init__(self, token_ids: list[int], vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def get_batch(self, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - block_size - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + block_size] for s in starts])
        y = torch.stack([self.data[s + 1 : s + block_size + 1] for s in starts])
        return x, y


def split_train_val_ids(ids: list[int], val_ratio: float = 0.1) -> tuple[list[int], list[int]]:
    split_at = int(len(ids) * (1.0 - val_ratio))
    return ids[:split_at], ids[split_at:]


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


class WordLM(nn.Module):
    """Tiny transformer that predicts the next word-level token."""

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
    model: WordLM,
    train_data: TokenDataset,
    val_data: TokenDataset,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_batches: int = 15,
) -> tuple[float, float]:
    model.eval()
    result: dict[str, float] = {}

    for name, dataset in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_batches):
            x, y = dataset.get_batch(batch_size, block_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)

    model.train()
    return result["train"], result["val"]


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    recent_ids: list[int],
    penalty: float = 1.5,
) -> torch.Tensor:
    """Down-weight tokens that appeared recently (discourages phrase loops)."""
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
    model: WordLM,
    tokenizer: WordTokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.9,
    top_k: int = 20,
    repetition_penalty: float = 1.5,
    repetition_window: int = 12,
    seed: int | None = None,
) -> str:
    """
    Sample next WORD tokens (not letters).

    Compared to phase 07 char-level:
      - loops look like "the the the" not "tttttt"
      - output is more readable even when wrong
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    prompt_tokens = tokenize(prompt)
    ids = tokenizer.encode([t for t in prompt_tokens if t in tokenizer.stoi])
    if not ids:
        ids = [1]

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

        # Stop at end-of-line token if model emits one after enough words
        if tokenizer.itos[next_id] == "\n" and len(generated) >= 8:
            break

    return tokenizer.decode(ids)


def main() -> None:
    print("Phase 08: Word-level language model")
    print("=" * 60)

    # --- Build corpus and tokenize ---
    raw_text = build_corpus(min_words=10_000)
    tokens = tokenize(raw_text)
    word_count = len(raw_text.split())

    tokenizer = WordTokenizer()
    tokenizer.build_vocab(tokens)
    all_ids = tokenizer.encode(tokens)
    train_ids, val_ids = split_train_val_ids(all_ids, val_ratio=0.1)

    train_set = TokenDataset(train_ids, tokenizer.vocab_size)
    val_set = TokenDataset(val_ids, tokenizer.vocab_size)

    # CPU-friendly hyperparameters
    block_size = 48
    d_model = 96
    num_heads = 4
    num_layers = 2
    d_ff = 192
    batch_size = 32
    learning_rate = 2e-3
    dropout = 0.2
    max_epochs = 250
    eval_every = 25

    device = torch.device("cpu")

    print(f"Corpus words (approx): {word_count:,}")
    print(f"Total tokens:          {len(tokens):,}")
    print(f"Vocab size:            {tokenizer.vocab_size:,}")
    print(f"Train tokens:          {len(train_ids):,}")
    print(f"Val tokens:            {len(val_ids):,}")
    print(f"Block size:            {block_size} words/tokens")
    print(f"Device:                {device}\n")

    model = WordLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    print(f"Model parameters:      {count_parameters(model):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_evals = 0

    model.train()
    for epoch in range(1, max_epochs + 1):
        x, y = train_set.get_batch(batch_size, block_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

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
                stale_evals = 0
            else:
                stale_evals += 1

            print(
                f"Epoch {epoch:4d} | train loss = {train_loss:.4f} | "
                f"val loss = {val_loss:.4f}"
            )

            if stale_evals >= 4:
                print(f"Early stop at epoch {epoch}: val loss plateaued.")
                break

    print(f"\nBest val loss: {best_val:.4f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Using best validation checkpoint for generation.\n")

    prompts = ["To be", "All the world's", "Friends,"]
    temperatures = [0.7, 1.0, 1.2]

    print("--- Word-level generation ---")
    print(
        "Compare to phase 07: you should see word/phrase repeats, not letter spam."
    )

    for temp in temperatures:
        print(f"\n=== temperature = {temp} | top_k = 20 ===")
        for prompt in prompts:
            text = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=35,
                temperature=temp,
                top_k=20,
                seed=42,
            )
            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  Output: \"{text}\"")

    print("\n" + "=" * 60)
    print("Char-level (07): repeats letters → 'rrrr', 'eeee'")
    print("Word-level (08): repeats words  → 'the the', 'and and'")
    print("Word-level output is more readable even on tiny data.")


if __name__ == "__main__":
    main()
