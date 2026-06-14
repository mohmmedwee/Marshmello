"""
Phase 07: Tiny character-level language model

We train a small transformer to predict the next character in a text corpus.

Character-level (not word-level) keeps the alphabet small and runs fast
on a laptop. Same idea as GPT: given context, predict P(next char).

Architecture:
  char embedding → positional encoding → 2 transformer blocks → linear head

WHY GENERATED TEXT REPEATS ON TINY DATA
---------------------------------------
With only a few thousand characters the model can nearly MEMORIZE the corpus.
Logits become sharp on a few characters ('e', 'r', 't'), so sampling loops.
We reduce this with dropout, early stopping, and decoding guards that block
runaway repetition before it fills the whole output string.
"""

import copy
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

All the world's a stage, and all the men and women merely players;
They have their exits and their entrances, and one man in his time
plays many parts, his acts being seven ages. At first the infant,
mewling and puking in the nurse's arms; then the whining school-boy,
with his satchel and shining morning face, creeping like snail
unwillingly to school. And then the lover, sighing like furnace,
with a woeful ballad made to his mistress' eyebrow.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.

Double, double toil and trouble;
Fire burn and caldron bubble.
By the pricking of my thumbs,
Something wicked this way comes.
Open, locks, whoever knocks!

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.

The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven
Upon the place beneath. It is twice blest;
It blesseth him that gives and him that takes.
'Tis mightiest in the mightiest; it becomes
The throned monarch better than his crown.

Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing. Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day.
""".strip()


class CharDataset:
    """Maps characters ↔ integer IDs and samples random (input, target) chunks."""

    def __init__(
        self,
        text: str,
        stoi: dict[str, int] | None = None,
        itos: dict[int, str] | None = None,
    ) -> None:
        if stoi is None or itos is None:
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.stoi = stoi
            self.itos = itos

        self.vocab_size = len(self.stoi)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def get_batch(self, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - block_size - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + block_size] for s in starts])
        y = torch.stack([self.data[s + 1 : s + block_size + 1] for s in starts])
        return x, y


def split_train_val(text: str, val_ratio: float = 0.1) -> tuple[str, str]:
    split_at = int(len(text) * (1.0 - val_ratio))
    return text[:split_at], text[split_at:]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_for_print(text: str, max_len: int = 220) -> str:
    shown = text.replace("\n", "\\n")
    if len(shown) > max_len:
        return shown[:max_len] + "..."
    return shown


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


class CharLM(nn.Module):
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


@torch.no_grad()
def estimate_loss(
    model: CharLM,
    train_data: CharDataset,
    val_data: CharDataset,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_batches: int = 20,
) -> tuple[float, float]:
    model.eval()
    losses: dict[str, float] = {}

    for name, dataset in [("train", train_data), ("val", val_data)]:
        batch_losses = []
        for _ in range(eval_batches):
            x, y = dataset.get_batch(batch_size, block_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, dataset.vocab_size),
                y.view(-1),
            )
            batch_losses.append(loss.item())
        losses[name] = sum(batch_losses) / len(batch_losses)

    model.train()
    return losses["train"], losses["val"]


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    top_values, top_indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, top_indices, top_values)
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    recent_ids: list[int],
    penalty: float = 1.8,
) -> torch.Tensor:
    """
    Frequency-aware penalty: chars that appeared MORE in the window get
    penalized MORE (penalty ** count), not just a flat divide.
    """
    adjusted = logits.clone()
    counts = Counter(recent_ids)
    for token_id, count in counts.items():
        factor = penalty**count
        if adjusted[token_id] > 0:
            adjusted[token_id] /= factor
        else:
            adjusted[token_id] *= factor
    return adjusted


def block_consecutive_char(logits: torch.Tensor, ids: list[int], max_run: int = 3) -> torch.Tensor:
    """
    Hard ban: if the last (max_run - 1) chars are identical, forbid that char next.
    Stops 'eee...' and 'rrr...' loops before they start.
    """
    if len(ids) < max_run - 1:
        return logits
    last_id = ids[-1]
    if all(ids[i] == last_id for i in range(-(max_run - 1), 0)):
        logits = logits.clone()
        logits[last_id] = float("-inf")
    return logits


def block_repeating_ngrams(logits: torch.Tensor, ids: list[int], ngram_size: int = 4) -> torch.Tensor:
    """
    Ban any token that would recreate an n-gram already seen in the context.
    Stops short loops like 'ererer' or 't t t t'.
    """
    if len(ids) < ngram_size - 1:
        return logits

    logits = logits.clone()
    prefix = tuple(ids[-(ngram_size - 1) :])
    for start in range(len(ids) - ngram_size + 1):
        ngram = tuple(ids[start : start + ngram_size])
        if ngram[:-1] == prefix:
            logits[ngram[-1]] = float("-inf")
    return logits


def block_newline_spam(
    logits: torch.Tensor,
    generated_ids: list[int],
    dataset: CharDataset,
    max_recent_newlines: int = 2,
    window: int = 8,
) -> torch.Tensor:
    """Corpus has many blank lines; prevent sampling newline after newline."""
    if "\n" not in dataset.stoi:
        return logits

    newline_id = dataset.stoi["\n"]
    recent = generated_ids[-window:]
    if recent.count(newline_id) >= max_recent_newlines:
        logits = logits.clone()
        logits[newline_id] = float("-inf")
    return logits


def trailing_repeat_count(ids: list[int], itos: dict[int, str]) -> int:
    if not ids:
        return 0
    last_char = itos[ids[-1]]
    count = 0
    for token_id in reversed(ids):
        if itos[token_id] == last_char:
            count += 1
        else:
            break
    return count


@torch.no_grad()
def generate(
    model: CharLM,
    dataset: CharDataset,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 10,
    repetition_penalty: float = 1.8,
    repetition_window: int = 40,
    max_repeat_stop: int = 4,
    seed: int | None = None,
) -> str:
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    prompt_ids = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    if not prompt_ids:
        prompt_ids = [0]

    ids = list(prompt_ids)
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.block_size :]], dtype=torch.long)
        logits = model(context)[0, -1, :].clone()

        recent = generated_ids[-repetition_window:]
        if recent:
            logits = apply_repetition_penalty(logits, recent, penalty=repetition_penalty)

        logits = block_consecutive_char(logits, ids, max_run=3)
        logits = block_repeating_ngrams(logits, ids, ngram_size=4)
        logits = block_newline_spam(logits, generated_ids, dataset)

        logits = logits / temperature
        logits = apply_top_k(logits, top_k)

        # Fallback if every token was banned (rare edge case)
        if not torch.isfinite(logits).any():
            logits = model(context)[0, -1, :] / temperature
            logits = apply_top_k(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        ids.append(next_id)
        generated_ids.append(next_id)

        if trailing_repeat_count(ids, dataset.itos) >= max_repeat_stop:
            break

    return "".join(dataset.itos[i] for i in ids)


def main() -> None:
    print("Phase 07: Tiny character-level language model")
    print("=" * 60)

    block_size = 64
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128
    batch_size = 32
    learning_rate = 3e-3
    dropout = 0.25
    max_epochs = 300
    val_ratio = 0.1
    eval_every = 25
    patience = 4
    max_overfit_gap = 0.45

    device = torch.device("cpu")

    train_text, val_text = split_train_val(TEXT, val_ratio=val_ratio)
    base_vocab = CharDataset(TEXT)
    train_set = CharDataset(train_text, stoi=base_vocab.stoi, itos=base_vocab.itos)
    val_set = CharDataset(val_text, stoi=base_vocab.stoi, itos=base_vocab.itos)

    print(f"Corpus length:   {len(TEXT):,} characters")
    print(f"Train chars:     {len(train_text):,}")
    print(f"Validation chars:{len(val_text):,}  ({val_ratio:.0%} held out)")
    print(f"Vocabulary:      {base_vocab.vocab_size} unique characters")
    print(f"Block size:      {block_size}")
    print(f"Dropout:         {dropout}")
    print(f"Max epochs:      {max_epochs}")
    print(f"Training on:     {device}\n")

    model = CharLM(
        vocab_size=base_vocab.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    evals_without_improvement = 0
    stopped_early = False
    stop_reason = ""

    model.train()
    for epoch in range(1, max_epochs + 1):
        x, y = train_set.get_batch(batch_size, block_size)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, base_vocab.vocab_size),
            y.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0 or epoch == 1:
            train_loss, val_loss = estimate_loss(
                model, train_set, val_set, block_size, batch_size, device
            )
            gap = val_loss - train_loss

            if epoch == 1 or epoch % 100 == 0 or epoch == max_epochs:
                print(
                    f"Epoch {epoch:4d} | train loss = {train_loss:.4f} | "
                    f"val loss = {val_loss:.4f} | gap = {gap:+.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if evals_without_improvement >= patience:
                stopped_early = True
                stop_reason = f"no val improvement for {patience} checks"
                print(f"Early stopping at epoch {epoch}: {stop_reason}.")
                break

            if epoch >= 75 and gap > max_overfit_gap:
                stopped_early = True
                stop_reason = f"train-val gap {gap:.2f} > {max_overfit_gap}"
                print(f"Early stopping at epoch {epoch}: {stop_reason}.")
                break

    print(
        f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}"
        + (f" ({stop_reason})" if stopped_early else "")
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best validation checkpoint for generation.\n")
    else:
        print("Warning: no best checkpoint saved; using final weights.\n")

    prompts = ["To be", "Whether", "All the"]
    temperatures = [0.7, 1.0, 1.3]

    print("--- Generation samples (best checkpoint) ---")
    print(
        "Guards: top_k=10, freq repetition penalty, 3-char run ban, "
        "4-gram block, newline cap, stop after 4 identical chars."
    )

    for temp in temperatures:
        print(f"\n=== temperature = {temp} | top_k = 10 ===")
        for prompt in prompts:
            sample = generate(
                model,
                base_vocab,
                prompt,
                max_new_tokens=200,
                temperature=temp,
                top_k=10,
                seed=42,
            )
            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  Output: \"{format_for_print(sample)}\"")

    print("\n" + "=" * 60)
    print("If output still looks odd: tiny corpus + char-level = expected.")
    print("The guards above prevent runaway 'eeee' / 'rrrr' loops.")


if __name__ == "__main__":
    main()
