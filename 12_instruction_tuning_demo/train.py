"""
Phase 12: Instruction tuning demo

PRETRAINING (phases 07–10)
--------------------------
The model learns P(next token | previous tokens) on raw text.
It might continue a paragraph — not necessarily answer a question.

INSTRUCTION TUNING (this phase)
-------------------------------
We still use next-token prediction, but the training text is formatted as chat:

    <USER> What is AI? <ASSISTANT> AI is a field of ... <END>

The model learns:
  1. WHEN it sees <ASSISTANT>, produce a helpful answer (not more user text)
  2. WHEN it sees <END>, stop
  3. The *style* of short educational responses in our tiny dataset

Real ChatGPT-style models do pretraining on trillions of tokens, THEN
instruction tuning / RLHF on curated dialog data. Same idea, smaller scale.
"""

from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Special format tokens (must appear in vocab as single tokens)
# ---------------------------------------------------------------------------
USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
END_TAG = "<END>"
PAD_TAG = "<PAD>"

ConceptExample = tuple[str, str, str]

OVERFIT_EXAMPLES: list[ConceptExample] = [
    ("What is AI?", "AI is artificial intelligence.", "AI"),
    ("What is BPE?", "BPE is a subword tokenizer.", "BPE"),
]


# ---------------------------------------------------------------------------
# Tiny embedded instruction dataset (question → answer pairs)
# ---------------------------------------------------------------------------
INSTRUCTION_EXAMPLES: list[tuple[str, str]] = [
    (
        "What is AI?",
        "AI is a field of computer science that builds systems "
        "which can learn patterns and make decisions from data.",
    ),
    (
        "Explain transformer simply.",
        "A transformer is a neural network that uses attention "
        "so each token can look at other tokens and build context.",
    ),
    (
        "What is backpropagation?",
        "Backpropagation computes gradients by applying the chain rule "
        "from the loss backward through each layer to update weights.",
    ),
    (
        "What is a neuron?",
        "A neuron computes a weighted sum of inputs plus a bias, "
        "then applies an activation function to produce an output.",
    ),
    (
        "What is loss?",
        "Loss is a number that measures how wrong the model's predictions are. "
        "Training tries to make loss smaller.",
    ),
    (
        "What is gradient descent?",
        "Gradient descent updates weights in the direction that reduces loss, "
        "taking small steps controlled by the learning rate.",
    ),
    (
        "What is attention?",
        "Attention lets each token create a query, match keys from other tokens, "
        "and blend their values into a new representation.",
    ),
    (
        "What is an embedding?",
        "An embedding maps a discrete token ID to a continuous vector "
        "so the model can do math on words or subwords.",
    ),
    (
        "What is BPE?",
        "BPE is a subword tokenizer that merges frequent character pairs "
        "so common words stay whole and rare words split into pieces.",
    ),
    (
        "What is overfitting?",
        "Overfitting happens when a model memorizes training data "
        "and performs poorly on new examples.",
    ),
    (
        "What is a validation set?",
        "A validation set is held-out data used to check generalization "
        "without updating model weights.",
    ),
    (
        "What is temperature in sampling?",
        "Temperature scales logits before softmax. Lower values make output "
        "more focused; higher values make it more random.",
    ),
    (
        "What is top-k sampling?",
        "Top-k sampling restricts random choice to the k most likely next tokens, "
        "reducing junk outputs.",
    ),
    (
        "What is a token?",
        "A token is the basic unit of text the model reads — "
        "a character, word, or subword depending on the tokenizer.",
    ),
    (
        "What is pretraining?",
        "Pretraining teaches a language model to predict the next token "
        "on large unstructured text corpora.",
    ),
    (
        "What is instruction tuning?",
        "Instruction tuning fine-tunes a pretrained model on formatted "
        "user-assistant examples so it follows prompts helpfully.",
    ),
    (
        "What is a transformer block?",
        "A block contains self-attention, a feed-forward network, "
        "residual connections, and layer normalization.",
    ),
    (
        "What is self-attention?",
        "Self-attention compares every token to every other token "
        "in the same sequence using queries, keys, and values.",
    ),
    (
        "What is layer norm?",
        "Layer normalization stabilizes training by scaling activations "
        "to have consistent mean and variance within each layer.",
    ),
    (
        "What is a residual connection?",
        "A residual connection adds the layer input to its output "
        "so gradients flow more easily through deep networks.",
    ),
    (
        "What is PyTorch?",
        "PyTorch is a Python library for building and training neural networks "
        "with automatic differentiation.",
    ),
    (
        "What is a language model?",
        "A language model assigns probabilities to sequences of tokens "
        "and can generate text one token at a time.",
    ),
    (
        "What is cross-entropy loss?",
        "Cross-entropy compares predicted token probabilities to the true next token. "
        "It is standard for classification and language modeling.",
    ),
    (
        "What is a learning rate?",
        "The learning rate controls how big each weight update step is during training.",
    ),
    (
        "What is dropout?",
        "Dropout randomly zeros neurons during training to reduce overfitting.",
    ),
    (
        "What is a prompt?",
        "A prompt is the input text you give a model to condition its generation.",
    ),
    (
        "What is fine-tuning?",
        "Fine-tuning continues training a pretrained model on a smaller "
        "specialized dataset for a new task.",
    ),
    (
        "What is GPT?",
        "GPT is a family of decoder-only transformer language models "
        "trained to predict the next token.",
    ),
    (
        "What is context length?",
        "Context length is the maximum number of tokens the model "
        "can attend to in one forward pass.",
    ),
    (
        "What is softmax?",
        "Softmax converts a vector of scores into probabilities "
        "that sum to one.",
    ),
]

CONCEPT_NAMES: list[str] = [
    "AI",
    "transformer",
    "backpropagation",
    "a neuron",
    "loss",
    "gradient descent",
    "attention",
    "an embedding",
    "BPE",
    "overfitting",
    "a validation set",
    "temperature in sampling",
    "top-k sampling",
    "a token",
    "pretraining",
    "instruction tuning",
    "a transformer block",
    "self-attention",
    "layer norm",
    "a residual connection",
    "PyTorch",
    "a language model",
    "cross-entropy loss",
    "a learning rate",
    "dropout",
    "a prompt",
    "fine-tuning",
    "GPT",
    "context length",
    "softmax",
]

QUESTION_PARAPHRASE_TEMPLATES: list[str] = [
    "What is {concept}?",
    "Explain {concept}.",
    "Define {concept}.",
    "{concept} meaning?",
    "Describe {concept}.",
    "Give a simple definition of {concept}.",
    "In simple terms, what is {concept}?",
    "What does {concept} mean?",
    "Can you explain {concept}?",
    "Tell me about {concept}.",
]


def expand_instruction_examples(
    examples: list[tuple[str, str]],
    concept_names: list[str],
) -> list[ConceptExample]:
    """Create 10 unique question paraphrases per concept for full mode."""
    if len(examples) != len(concept_names):
        raise ValueError("Each instruction example needs exactly one concept name.")

    expanded: list[ConceptExample] = []
    for (canonical_question, answer), concept in zip(examples, concept_names):
        questions: list[str] = [canonical_question]
        for template in QUESTION_PARAPHRASE_TEMPLATES:
            question = template.format(concept=concept)
            if question not in questions:
                questions.append(question)
            if len(questions) == 10:
                break

        for question in questions:
            expanded.append((question, answer, concept))

    return expanded


def split_unique_questions_by_concept(
    examples: list[ConceptExample],
) -> tuple[list[ConceptExample], list[ConceptExample]]:
    """
    Hold out one unique paraphrase per concept before any training repeats.

    Validation should test unseen questions, not later copies of repeated
    training strings.
    """
    grouped: dict[str, list[ConceptExample]] = {}
    for example in examples:
        grouped.setdefault(example[2], []).append(example)

    train: list[ConceptExample] = []
    val: list[ConceptExample] = []
    for concept_examples in grouped.values():
        train.extend(concept_examples[:-1])
        val.append(concept_examples[-1])

    return train, val


def format_chat(user: str, assistant: str) -> str:
    """Standard chat format used for instruction tuning."""
    return f"{USER_TAG} {user} {ASSISTANT_TAG} {assistant} {END_TAG}"


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer (special tags are single tokens)."""
    return text.split()


class ChatTokenizer:
    """Maps chat tokens ↔ integer IDs."""

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def build_vocab(self, texts: list[str]) -> None:
        if PAD_TAG not in self.stoi:
            self.stoi[PAD_TAG] = len(self.stoi)
            self.itos[self.stoi[PAD_TAG]] = PAD_TAG
        for text in texts:
            for token in tokenize(text):
                if token not in self.stoi:
                    idx = len(self.stoi)
                    self.stoi[token] = idx
                    self.itos[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[t] for t in tokenize(text) if t in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids)


class TokenDataset:
    """
    Sample random windows from a long token stream (phases 07–10 style).

    For instruction tuning we prefer ExampleDataset so a batch window
    stays inside ONE chat example (see below).
    """

    def __init__(self, token_ids: list[int], vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def get_batch(self, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - block_size - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + block_size] for s in starts])
        y = torch.stack([self.data[s + 1 : s + block_size + 1] for s in starts])
        return x, y


class ExampleDataset:
    """
    Each instruction example is kept separate during training.

    Random slices never cross <USER>…<END> boundaries, so the model
    learns coherent question→answer patterns instead of chopped fragments.
    """

    def __init__(
        self,
        examples: list[list[int]],
        vocab_size: int,
        block_size: int,
        pad_id: int | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.pad_id = pad_id
        self.examples: list[list[int]] = []
        for ex in examples:
            if len(ex) >= block_size + 1:
                self.examples.append(ex)
            elif pad_id is not None and len(ex) >= 2:
                self.examples.append(ex + [pad_id] * (block_size + 1 - len(ex)))

    def get_batch(self, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(batch_size):
            ex = self.examples[torch.randint(0, len(self.examples), (1,)).item()]
            start = torch.randint(0, len(ex) - block_size, (1,)).item()
            chunk = ex[start : start + block_size + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.tensor(xs), torch.tensor(ys)


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


def build_causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build a GPT-style causal mask for decoder self-attention.

    Shape: (seq_len, seq_len)
    Values: 0 on and below diagonal, -inf ABOVE diagonal.

    Token at row i may attend to columns j <= i only (past + present).
    Positions j > i are masked out (cannot peek at future tokens).

    Encoder attention (phase 06) is bidirectional — all tokens see each other.
    Decoder / language modeling MUST be causal — otherwise training cheats by
    letting position t read the correct answer at t+1, giving fake low loss
    but broken generation. GPT is decoder-only with causal self-attention.
    """
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
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
        seq_len = x.size(1)
        # Causal mask: (seq_len, seq_len), -inf above diagonal → no future peeking
        causal_mask = build_causal_attention_mask(seq_len, x.device)

        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask)
        x = x + self.dropout1(h)
        h = self.norm2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


class ChatLM(nn.Module):
    """
    Tiny decoder-only transformer for instruction-formatted text.

    Uses causal self-attention in every block (GPT-style), NOT bidirectional
    encoder attention from phase 06.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        block_size: int,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_emb = PositionalEncoding(d_model, block_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
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
    model: ChatLM,
    train_data: ExampleDataset,
    val_data: ExampleDataset,
    block_size: int,
    batch_size: int,
    device: torch.device,
    tokenizer: ChatTokenizer,
    batches: int = 12,
) -> tuple[float, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, dataset in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(batches):
            x, y = dataset.get_batch(batch_size, block_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            weights = build_assistant_loss_weights(x, y, tokenizer).to(device)
            losses.append(weighted_cross_entropy(logits, y, weights).item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out["train"], out["val"]


def build_assistant_loss_weights(
    x: torch.Tensor,
    targets: torch.Tensor,
    tokenizer: ChatTokenizer,
) -> torch.Tensor:
    """
    Weight SFT loss on assistant targets only.

    Real instruction tuning (SFT) typically ignores loss on the user prompt
    and trains only on the assistant's reply (+ <END>).

    Most answer continuation tokens are easy to predict from earlier answer
    tokens, so low average loss can hide bad routing from question to answer.
    The first target after <ASSISTANT> is the routing token, so it receives
    extra weight. <END> is also weighted so the model learns to stop.
    """
    assistant_id = tokenizer.stoi[ASSISTANT_TAG]
    end_id = tokenizer.stoi[END_TAG]
    pad_id = tokenizer.stoi[PAD_TAG]
    batch_size, seq_len = x.shape
    weights = torch.zeros(batch_size, seq_len, dtype=torch.float)

    for row in range(batch_size):
        ids = x[row].tolist()
        try:
            start = ids.index(assistant_id)
        except ValueError:
            continue

        for i in range(start, seq_len):
            target_id = int(targets[row, i].item())
            if target_id == pad_id:
                break
            if i == start:
                # x[start] is <ASSISTANT>; target[start] is the first answer token.
                weights[row, i] = 8.0
            elif target_id == end_id:
                weights[row, i] = 2.0
                break
            else:
                weights[row, i] = 1.0

    return weights


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy averaged over weighted assistant target positions."""
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
    loss = loss.view(weights.shape)
    weighted = loss * weights
    return weighted.sum() / weights.sum().clamp(min=1.0)


def _full_sequence_assistant_idx(example_ids: list[int], assistant_id: int) -> int:
    try:
        return example_ids.index(assistant_id)
    except ValueError:
        return len(example_ids)


def _verify_end_target_in_loss(
    example_ids: list[int],
    assistant_id: int,
    end_id: int,
    tokenizer: ChatTokenizer,
) -> tuple[bool, int | None, str | None]:
    """
    Check that predicting <END> from the prior token is included in SFT loss.

    Uses the full example (not a truncated window) because block_size may end
    exactly one token before <END> when start=0.
    """
    assistant_idx = _full_sequence_assistant_idx(example_ids, assistant_id)
    for i in range(len(example_ids) - 1):
        if example_ids[i + 1] == end_id:
            in_loss = i >= assistant_idx
            before = tokenizer.itos[example_ids[i]]
            return in_loss, i, before
    return False, None, None


def debug_sft_loss_sample(
    tokenizer: ChatTokenizer,
    example_ids: list[int],
    block_size: int,
) -> None:
    """
    Print how the weighted SFT loss treats one training example.

    At position i the model reads x[i] and is trained to predict y[i] = x[i+1]
    (next-token prediction). weight[i]>0 means that prediction contributes to loss.

    Expected behavior:
      - <USER> and question tokens → weight=0 (ignored)
      - <ASSISTANT> position → weight=8 (learn first answer/routing token)
      - answer tokens → weight=1
      - position before <END> → weight=2 (learn to emit <END>)
    """
    chunk = example_ids[: block_size + 1]
    if len(chunk) < 2:
        print("  (example too short for debug)\n")
        return

    x = torch.tensor([chunk[:-1]], dtype=torch.long)
    y = torch.tensor([chunk[1:]], dtype=torch.long)
    weights = build_assistant_loss_weights(x, y, tokenizer)

    x_ids = x[0].tolist()
    y_ids = y[0].tolist()
    w = weights[0].tolist()

    assistant_id = tokenizer.stoi[ASSISTANT_TAG]
    user_id = tokenizer.stoi[USER_TAG]
    end_id = tokenizer.stoi[END_TAG]

    try:
        assistant_idx = x_ids.index(assistant_id)
    except ValueError:
        assistant_idx = len(x_ids)

    assistant_weight = float(weights.sum().item())
    user_tokens_ignored = sum(1 for i in range(assistant_idx) if w[i] == 0.0)

    print("--- Weighted SFT loss debug (first training sample) ---")
    print(f"  Assistant loss weight total:    {assistant_weight:.1f}")
    print(f"  User tokens ignored:            {user_tokens_ignored}")
    if len(example_ids) > block_size + 1:
        print(
            f"  (Window shows first {block_size} input tokens; "
            f"example has {len(example_ids)} total.)"
        )
    print()

    vis = " ".join(f"{v:g}" for v in w)
    print(f"  Weight visualization (0=ignore): {vis}")
    print()

    user_positions = [i for i, tid in enumerate(x_ids) if tid == user_id]
    user_all_ignored = all(w[i] == 0.0 for i in user_positions)
    assistant_pos_weighted = assistant_idx < len(w) and w[assistant_idx] == 8.0
    end_ok, end_full_idx, token_before_end = _verify_end_target_in_loss(
        example_ids, assistant_id, end_id, tokenizer
    )

    print("  Verification:")
    print(f"    <USER> tokens ignored:           {user_all_ignored}")
    print(f"    first answer token weight=8:     {assistant_pos_weighted}")
    print(f"    <END> target contributes:        {end_ok}")
    if end_full_idx is not None and token_before_end is not None:
        print(
            f"      (full index {end_full_idx}: "
            f'"{token_before_end}" → <END>, in loss={end_ok})'
        )
    print()

    print("  token-by-token (input → predict next):")
    print(f"  {'token':<16} | {'weight':<8} | target token")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*16}")
    for i in range(len(x_ids)):
        token = tokenizer.itos[x_ids[i]]
        target = tokenizer.itos[y_ids[i]]
        print(f"  {token:<16} | {w[i]:<8g} | {target}")

    end_in_window = end_id in y_ids
    if not end_in_window and end_full_idx is not None and token_before_end is not None:
        print(
            f"  {'...':<16} | {'(truncated)':<8} | "
            f'<END> at full index {end_full_idx} (in loss={"yes" if end_ok else "no"})'
        )
    print()


def print_sft_loss_summary(
    tokenizer: ChatTokenizer,
    examples: list[list[int]],
    block_size: int,
    max_examples: int = 50,
) -> None:
    """Aggregate mask stats over several training windows (from example starts)."""
    total_weight = 0.0
    total_ignored = 0
    total_user_ignored = 0
    assistant_id = tokenizer.stoi[ASSISTANT_TAG]

    for ex in examples[:max_examples]:
        chunk = ex[: block_size + 1]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long)
        y = torch.tensor([chunk[1:]], dtype=torch.long)
        weights = build_assistant_loss_weights(x, y, tokenizer)[0].tolist()
        x_ids = x[0].tolist()
        try:
            assistant_idx = x_ids.index(assistant_id)
        except ValueError:
            assistant_idx = len(x_ids)

        total_weight += sum(weights)
        total_ignored += sum(1 for v in weights if v == 0.0)
        total_user_ignored += sum(1 for i in range(assistant_idx) if weights[i] == 0.0)

    n = min(max_examples, len(examples))
    print("--- SFT loss summary (first example windows from start) ---")
    print(f"  Averaged over {n} examples:")
    print(f"    avg assistant loss weight:    {total_weight / n:.1f}")
    print(f"    avg ignored positions:        {total_ignored / n:.1f}")
    print(f"    avg user-region ignored:      {total_user_ignored / n:.1f}")
    print()


def build_training_corpus(examples: list[ConceptExample], repeats: int = 40) -> list[str]:
    """
    Repeat formatted chats so the tiny model sees enough signal.
    Real instruction tuning uses thousands–millions of examples once;
    we repeat a small set for demo purposes only.
    """
    formatted = [format_chat(question, answer) for question, answer, _ in examples]
    return formatted * repeats


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return filtered


def has_repeated_ngram(token_ids: list[int], n: int = 5) -> bool:
    """
    Return True if any n-token sequence appears twice in token_ids.
    Stops runaway loops like 'the tokenizer. the tokenizer.'
    """
    if len(token_ids) < 2 * n:
        return False
    seen: set[tuple[int, ...]] = set()
    for i in range(len(token_ids) - n + 1):
        ngram = tuple(token_ids[i : i + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def first_n_word_tokens(text: str, n: int = 5) -> list[str]:
    """First n whitespace tokens from an answer string."""
    return tokenize(text)[:n]


def normalize_answer(text: str) -> str:
    """Whitespace-normalize generated and expected answers before exact checks."""
    return " ".join(tokenize(text))


def concept_signatures(examples: list[ConceptExample]) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Map each concept to first-token and first-5-token answer signatures."""
    signatures: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for _, answer, concept in examples:
        signatures.setdefault(
            concept,
            (tuple(first_n_word_tokens(answer, 1)), tuple(first_n_word_tokens(answer, 5))),
        )
    return signatures


def predict_concept_from_answer(
    generated: str,
    signatures: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
) -> str | None:
    """Infer the routed concept from generated answer tokens."""
    generated_first = tuple(first_n_word_tokens(generated, 1))
    generated_five = tuple(first_n_word_tokens(generated, 5))

    for concept, (_, first_five) in signatures.items():
        if generated_five == first_five:
            return concept

    first_token_matches = [
        concept
        for concept, (first_token, _) in signatures.items()
        if generated_first == first_token
    ]
    if len(first_token_matches) == 1:
        return first_token_matches[0]
    return None


@torch.no_grad()
def generate_assistant_reply(
    model: ChatLM,
    tokenizer: ChatTokenizer,
    user_prompt: str,
    max_answer_tokens: int = 35,
    greedy: bool = True,
    temperature: float = 0.8,
    top_k: int = 20,
    ngram_stop: int = 5,
    seed: int | None = None,
) -> str:
    """
    Build prompt ending with <ASSISTANT>, then generate the answer.

    greedy=True (default for demos):
      Pick argmax each step — less random mixing of memorized answers.

    Stops when:
      - <END> token is produced
      - max_answer_tokens reached (default 35)
      - the same 5-token sequence repeats (repetition guard)
    """
    model.eval()
    if seed is not None and not greedy:
        torch.manual_seed(seed)

    end_id = tokenizer.stoi[END_TAG]
    banned_ids = {
        tokenizer.stoi[USER_TAG],
        tokenizer.stoi[ASSISTANT_TAG],
        tokenizer.stoi[PAD_TAG],
    }

    prefix = f"{USER_TAG} {user_prompt} {ASSISTANT_TAG}"
    ids = tokenizer.encode(prefix)
    answer_ids: list[int] = []  # tokens generated AFTER <ASSISTANT>

    for _ in range(max_answer_tokens):
        context = torch.tensor([ids[-model.block_size :]], dtype=torch.long)
        logits = model(context)[0, -1, :].clone()

        for bid in banned_ids:
            logits[bid] = float("-inf")

        if greedy:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            logits = apply_top_k(logits, top_k)
            if not torch.isfinite(logits).any():
                break
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == end_id:
            break

        ids.append(next_id)
        answer_ids.append(next_id)

        if has_repeated_ngram(answer_ids, n=ngram_stop):
            break

    full_text = tokenizer.decode(ids)
    answer = full_text.split(ASSISTANT_TAG, 1)[-1]
    for tag in (END_TAG, USER_TAG, ASSISTANT_TAG, PAD_TAG):
        answer = answer.replace(tag, "")
    return answer.strip()


def evaluate_routing_accuracy(
    model: ChatLM,
    tokenizer: ChatTokenizer,
    examples: list[ConceptExample],
    label: str,
    signature_examples: list[ConceptExample],
    print_details: bool = True,
    max_details: int = 20,
) -> tuple[int, int, int, int]:
    """
    Greedy-generate and check whether each question routes to the right answer.

    First-token accuracy measures the branch from question to answer. First-5
    accuracy catches frequent-answer collapse where the first word happens to
    match but the answer quickly turns into a different memorized reply.
    """
    if print_details:
        print(f"--- Routing accuracy ({label}, greedy) ---")
        print("Checks if each question routes to the right memorized answer.\n")

    first_token_matches = 0
    first_five_matches = 0
    exact_matches = 0
    concept_matches = 0
    details_printed = 0
    details_skipped = 0
    signatures = concept_signatures(signature_examples)
    for question, expected, expected_concept in examples:
        generated = generate_assistant_reply(
            model, tokenizer, question, greedy=True
        )
        exp_first = first_n_word_tokens(expected, 1)
        gen_first = first_n_word_tokens(generated, 1)
        exp_five = first_n_word_tokens(expected, 5)
        gen_five = first_n_word_tokens(generated, 5)

        first_ok = gen_first == exp_first
        five_ok = gen_five == exp_five
        exact_ok = normalize_answer(generated) == normalize_answer(expected)
        predicted_concept = predict_concept_from_answer(generated, signatures)
        concept_ok = predicted_concept == expected_concept

        first_token_matches += int(first_ok)
        first_five_matches += int(five_ok)
        exact_matches += int(exact_ok)
        concept_matches += int(concept_ok)

        show_detail = print_details and (
            details_printed < max_details or not five_ok or not concept_ok
        )
        if show_detail:
            status = "MATCH" if five_ok else "MISS"
            print(f"  [{status}] Q: {question}")
            print(f"         expected first token: {exp_first}")
            print(f"         generated first token: {gen_first}")
            print(f"         expected first 5:     {exp_five}")
            print(f"         generated first 5:    {gen_five}")
            print(f"         concept:              {predicted_concept} / {expected_concept}")
            if not exact_ok:
                print(f"         full generated:       {generated[:100]}...")
            print()
            details_printed += 1
        elif print_details:
            details_skipped += 1

    total = len(examples)
    first_pct = 100.0 * first_token_matches / total
    five_pct = 100.0 * first_five_matches / total
    exact_pct = 100.0 * exact_matches / total
    concept_pct = 100.0 * concept_matches / total
    if print_details:
        if details_skipped:
            print(f"  ... skipped {details_skipped} matching detail rows ...\n")
        print(
            f"  First-token routing accuracy: {first_token_matches}/{total} "
            f"({first_pct:.0f}%)"
        )
        print(
            f"  First-5-token routing accuracy: {first_five_matches}/{total} "
            f"({five_pct:.0f}%)"
        )
        print(f"  Exact answer match: {exact_matches}/{total} ({exact_pct:.0f}%)")
        print(f"  Concept accuracy: {concept_matches}/{total} ({concept_pct:.0f}%)\n")
    return first_token_matches, first_five_matches, exact_matches, concept_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 12 instruction tuning demo.")
    parser.add_argument(
        "--mode",
        choices=("full", "overfit"),
        default="full",
        help="full trains the whole demo split; overfit trains only two routing examples.",
    )
    return parser.parse_args()


def choose_block_size(encoded_examples: list[list[int]], desired: int) -> int:
    """
    Keep fixed-length batching, but avoid dropping short examples.

    ExampleDataset needs each example to provide block_size input tokens plus
    one target token. Overfit mode deliberately uses short answers, so the
    usable block size must be capped by the shortest selected chat.
    """
    shortest = min(len(ex) for ex in encoded_examples)
    return max(2, min(desired, shortest - 1))


def main() -> None:
    args = parse_args()
    print("Phase 12: Instruction tuning demo")
    print("=" * 60)
    print()
    print(f"Mode: {args.mode}")
    print("Pretraining  → predict next token on raw text (phases 07–10)")
    print("Instruction tuning → next-token loss on <USER>…<ASSISTANT>…<END> chats")
    print("Loss is weighted: assistant tokens only, with extra weight on routing.")
    print("Using causal decoder mask: True")
    print()

    if args.mode == "overfit":
        base_examples = OVERFIT_EXAMPLES
        train_base_examples = base_examples
        val_base_examples = base_examples
        signature_examples = base_examples
        repeats = 80
        desired_block_size = 26
        d_model = 96
        num_heads = 4
        num_layers = 2
        d_ff = 192
        batch_size = 16
        learning_rate = 4e-3
        max_epochs = 1200
        eval_every = 25
        dropout = 0.0
        weight_decay = 0.0
    else:
        base_examples = expand_instruction_examples(INSTRUCTION_EXAMPLES, CONCEPT_NAMES)
        train_base_examples, val_base_examples = split_unique_questions_by_concept(base_examples)
        signature_examples = base_examples
        repeats = 4
        desired_block_size = 64
        d_model = 192
        num_heads = 6
        num_layers = 4
        d_ff = 768
        batch_size = 24
        learning_rate = 2e-3
        max_epochs = 160
        eval_every = 20
        dropout = 0.15
        weight_decay = 0.01

    num_examples = len(base_examples)
    train_formatted = build_training_corpus(train_base_examples, repeats=repeats)
    val_repeats = 1 if args.mode == "full" else repeats
    val_formatted = build_training_corpus(val_base_examples, repeats=val_repeats)

    tokenizer = ChatTokenizer()
    tokenizer.build_vocab(train_formatted + val_formatted)

    train_examples = [tokenizer.encode(t) for t in train_formatted]
    val_examples = [tokenizer.encode(t) for t in val_formatted]

    if args.mode == "overfit":
        block_size = choose_block_size(train_examples + val_examples, desired_block_size)
    else:
        block_size = desired_block_size
    pad_id = tokenizer.stoi[PAD_TAG]
    train_set = ExampleDataset(train_examples, tokenizer.vocab_size, block_size, pad_id=pad_id)
    val_set = ExampleDataset(val_examples, tokenizer.vocab_size, block_size, pad_id=pad_id)

    if not train_set.examples:
        raise RuntimeError(
            f"No training examples longer than block_size={block_size}. "
            "Reduce block_size or shorten answers."
        )

    total_train_tokens = sum(len(ex) for ex in train_examples)
    total_val_tokens = sum(len(ex) for ex in val_examples)
    device = torch.device("cpu")

    print(f"Instruction examples:  {num_examples}")
    print(f"Unique train examples: {len(train_base_examples)}")
    print(f"Unique val examples:   {len(val_base_examples)}")
    print(f"Training chats (×{repeats}):  {len(train_formatted)}")
    print(f"Validation chats (×{val_repeats}): {len(val_formatted)}")
    print(f"Vocab size:            {tokenizer.vocab_size}")
    print(f"Block size:            {block_size}")
    print(f"Train tokens:          {total_train_tokens:,}")
    print(f"Val tokens:            {total_val_tokens:,}")

    model = ChatLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    print(f"Model parameters:      {count_parameters(model):,}\n")

    # --- SFT loss mask debugging (before training) ---
    first_example = train_set.examples[0]
    first_text = tokenizer.decode(first_example)
    print(f"First training sample (text):\n  {first_text}\n")
    debug_sft_loss_sample(tokenizer, first_example, block_size)
    print_sft_loss_summary(tokenizer, train_set.examples, block_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    reached_overfit = False
    train_check_examples = (
        train_base_examples if args.mode == "overfit" else train_base_examples[:30]
    )

    model.train()
    for epoch in range(1, max_epochs + 1):
        x, y = train_set.get_batch(batch_size, block_size)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        weights = build_assistant_loss_weights(x, y, tokenizer).to(device)
        loss = weighted_cross_entropy(logits, y, weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0 or epoch == 1:
            train_loss, val_loss = estimate_loss(
                model, train_set, val_set, block_size, batch_size, device, tokenizer
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            _, _, exact_matches, concept_matches = evaluate_routing_accuracy(
                model,
                tokenizer,
                train_check_examples,
                label="train",
                signature_examples=signature_examples,
                print_details=False,
            )
            print(
                f"Epoch {epoch:4d} | train loss = {train_loss:.4f} | "
                f"val loss = {val_loss:.4f} | exact = "
                f"{exact_matches}/{len(train_check_examples)} | concept = "
                f"{concept_matches}/{len(train_check_examples)}"
            )
            if args.mode == "overfit" and exact_matches == len(train_check_examples):
                best_state = copy.deepcopy(model.state_dict())
                reached_overfit = True
                print(f"Overfit exact match reached at epoch {epoch}: 2/2")
                break

    if args.mode == "overfit" and best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nLowest val loss observed: {best_val:.4f}")
    print(
        "(With causal masking, loss is often HIGHER than bidirectional cheating — "
        "that is expected and healthier for generation.)"
    )

    print("\n" + "=" * 60)
    print("Before causal mask: bidirectional attention → very low loss, wrong replies.")
    print("After causal mask:  each token only sees the past → harder training,")
    print("but generation matches how the model is used at inference time (GPT-style).")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("NOTE: Low loss on repeated tiny data does not prove question routing works.")
    print("Most answer continuation tokens are predictable after the answer starts,")
    print("so a model can reduce loss while still choosing the wrong first answer token.")
    print("Real instruction tuning needs")
    print("diverse dialog data, a stronger pretrained base, and more parameters.")
    print("=" * 60 + "\n")

    # --- Evaluate routing questions (greedy) ---
    evaluate_routing_accuracy(
        model,
        tokenizer,
        train_base_examples,
        label="train",
        signature_examples=signature_examples,
    )
    if args.mode == "full":
        evaluate_routing_accuracy(
            model,
            tokenizer,
            val_base_examples,
            label="validation",
            signature_examples=signature_examples,
        )
    elif not reached_overfit:
        print("WARNING: overfit mode ended before exact match reached 2/2.\n")

    print("--- Demo replies (greedy decoding, max 35 answer tokens) ---")
    print("Greedy = argmax each step. Stops at <END> or repeated 5-gram.\n")

    if args.mode == "overfit":
        demo_prompts = [question for question, _, _ in OVERFIT_EXAMPLES]
    else:
        demo_prompts = [
            "What is AI?",
            "Explain transformer simply.",
            "What is backpropagation?",
            "AI meaning?"
        ]

    gold_answers = {question: answer for question, answer, _ in base_examples}

    for prompt in demo_prompts:
        reply = generate_assistant_reply(model, tokenizer, prompt, greedy=True)
        print(f"User: {prompt}")
        if prompt in gold_answers:
            print(f"  Trained target: {gold_answers[prompt]}")
        print(f"  Generated:      {reply}\n")

    print("=" * 60)
    print("Instruction tuning did NOT change the architecture.")
    print("It changed the *training data format* so the model learns to act")
    print("like an assistant when it sees <USER> … <ASSISTANT>.")
    print("This demo also weights routing-sensitive answer tokens in the SFT loss.")
    print("Use greedy decoding for demos; sampling adds cross-answer mixing.")


if __name__ == "__main__":
    main()
