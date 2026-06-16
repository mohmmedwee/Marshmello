#!/usr/bin/env python3
"""
Phase 18B: SFT fine-tuning for Marshmello-45M-Instruct.

Loads Marshmello-45M-Base-v2 (large_50m checkpoint), trains only on assistant
response targets in Phase 17 chat JSONL, and writes instruct checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from config import GPTConfig, latest_checkpoint_for, resolve_config  # noqa: E402
from model.gpt import GPT, count_parameters  # noqa: E402
from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from tokenizer.decode import decode_ids_pretty  # noqa: E402
from tokenizer.encode import corpus_to_ids_safe  # noqa: E402
from training.trainer import pick_device  # noqa: E402

TOKENIZER_PATH = PHASE13_ROOT / "tokenizer" / "tokenizer.json"
CHAT_DATA_PATH = PROJECT_ROOT / "17_instruction_dataset" / "processed" / "chat.jsonl"
CHECKPOINT_DIR = PHASE_ROOT / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
CURATED_LATEST_CHECKPOINT = CHECKPOINT_DIR / "curated_latest.pt"

USER_TAG = "<USER>"
ASSISTANT_TAG = "<ASSISTANT>"
END_TAG = "<END>"
SFT_DEFAULT_LR = 1e-5
CURATED_DEFAULT_LR = 2e-6
SFT_GRAD_CLIP = 1.0
CURATED_MAX_EXAMPLES = 2_000
CURATED_MIN_INSTRUCTION_WORDS = 4
CURATED_MAX_INSTRUCTION_WORDS = 25
CURATED_MIN_RESPONSE_WORDS = 20
CURATED_MAX_RESPONSE_WORDS = 90
CURATED_BUCKETS = ("ai", "databases", "software_engineering", "cybersecurity_general")
CURATED_TECHNICAL_KEYWORDS = (
    "ai",
    "artificial",
    "intelligence",
    "machine",
    "learning",
    "model",
    "token",
    "transformer",
    "algorithm",
    "data",
    "database",
    "index",
    "sql",
    "query",
    "python",
    "function",
    "code",
    "api",
    "software",
    "engineering",
    "debug",
    "test",
    "security",
    "cybersecurity",
    "network",
    "encryption",
    "docker",
)
GENERIC_INSTRUCTION_PREFIXES = (
    "describe a time",
    "give three tips",
    "write a story",
    "write a poem",
    "generate a poem",
    "create a slogan",
    "make a list",
    "name three",
    "translate",
    "edit the following",
    "classify the following",
)
GENERATION_EVAL_TEMPERATURE = 0.2
GENERATION_EVAL_TOP_K = 10
GENERATION_EVAL_REPETITION_PENALTY = 1.3
GENERATION_EVAL_PROMPTS = (
    "What is AI?",
    "Explain database indexes.",
)
TRANSFORMER_KEY_PREFIXES = (
    "token_emb.",
    "pos_emb.",
    "blocks.",
    "norm.",
    "lm_head.",
)


@dataclass(frozen=True)
class ChatExample:
    text: str
    instruction: str
    response: str
    domain: str = "unknown"
    source: str = "unknown"


def find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for i in range(0, len(values) - len(needle) + 1):
        if values[i : i + len(needle)] == needle:
            return i
    return None


def encode_text(bpe, text: str) -> list[int]:
    """Encode SFT text, stripping characters missing from the BPE vocab."""
    return corpus_to_ids_safe(bpe, text)


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_chat_prompt(instruction: str) -> str:
    return f"{USER_TAG} {collapse_spaces(instruction)} {ASSISTANT_TAG}"


def format_sft_text(example: ChatExample) -> str:
    return f"{format_chat_prompt(example.instruction)} {collapse_spaces(example.response)} {END_TAG}"


def one_line(text: str, max_chars: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def normalize_exact(text: str) -> str:
    """Whitespace-normalize text for strict answer matching diagnostics."""
    return re.sub(r"\s+", " ", text).strip()


def parse_chat_text(text: str) -> tuple[str, str]:
    user_start = text.find(USER_TAG)
    assistant_start = text.find(ASSISTANT_TAG)
    end_start = text.rfind(END_TAG)
    if user_start < 0 or assistant_start < 0 or end_start < 0:
        raise ValueError(f"Chat example is missing one of {USER_TAG}, {ASSISTANT_TAG}, {END_TAG}")
    if not (user_start < assistant_start < end_start):
        raise ValueError("Chat example tags are not in <USER> ... <ASSISTANT> ... <END> order")
    instruction = text[user_start + len(USER_TAG) : assistant_start].strip()
    response = text[assistant_start + len(ASSISTANT_TAG) : end_start].strip()
    if not instruction or not response:
        raise ValueError("Chat example has empty instruction or response")
    return instruction, response


class CachedBPEEncoder:
    """Small word-level cache around the educational BPE encoder."""

    def __init__(self, bpe) -> None:
        self.bpe = bpe
        self.cache: dict[str, list[int]] = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            cached = self.cache.get(word)
            if cached is None:
                cached = corpus_to_ids_safe(self.bpe, word)
                self.cache[word] = cached
            ids.extend(cached)
        return ids


class SFTDataset:
    """Fixed-length SFT examples with assistant-only loss weights."""

    def __init__(
        self,
        texts: list[str],
        bpe,
        block_size: int,
        *,
        first_token_weight: float = 8.0,
        end_weight: float = 2.0,
        pad_id: int = 0,
    ) -> None:
        self.block_size = block_size
        self.pad_id = pad_id
        self.vocab_size = bpe.vocab_size
        encoder = CachedBPEEncoder(bpe)
        self.assistant_ids = encoder.encode(ASSISTANT_TAG)
        self.end_ids = encoder.encode(END_TAG)
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for text in texts:
            ids = encoder.encode(text)
            if len(ids) < 2:
                continue
            ids = ids[: block_size + 1]
            weights = self._weights_for_ids(
                ids,
                first_token_weight=first_token_weight,
                end_weight=end_weight,
            )
            if not any(weights):
                continue
            if len(ids) < block_size + 1:
                pad = [pad_id] * (block_size + 1 - len(ids))
                ids = ids + pad
                weights = weights + [0.0] * (block_size - len(weights))

            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            w = torch.tensor(weights[:block_size], dtype=torch.float)
            self.examples.append((x, y, w))

        if not self.examples:
            raise ValueError("No SFT examples produced non-empty assistant loss weights")

    def _weights_for_ids(
        self,
        ids: list[int],
        *,
        first_token_weight: float,
        end_weight: float,
    ) -> list[float]:
        weights = [0.0] * max(0, len(ids) - 1)
        assistant_start = find_subsequence(ids, self.assistant_ids)
        if assistant_start is None:
            return weights
        first_answer_target_pos = assistant_start + len(self.assistant_ids) - 1
        end_start = find_subsequence(ids, self.end_ids)
        end_last_target_pos = len(weights) - 1
        if end_start is not None:
            end_last_target_pos = min(len(weights) - 1, end_start + len(self.end_ids) - 2)

        for pos in range(first_answer_target_pos, end_last_target_pos + 1):
            weights[pos] = 1.0
        if 0 <= first_answer_target_pos < len(weights):
            weights[first_answer_target_pos] = first_token_weight
        if end_start is not None:
            for pos in range(max(first_answer_target_pos, end_start - 1), end_last_target_pos + 1):
                weights[pos] = end_weight
        return weights

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.examples), (batch_size,))
        xs, ys, ws = zip(*(self.examples[int(i)] for i in idx))
        return torch.stack(xs).to(device), torch.stack(ys).to(device), torch.stack(ws).to(device)


def load_chat_examples(path: Path) -> list[ChatExample]:
    examples: list[ChatExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            if text:
                instruction = str(record.get("instruction", "")).strip()
                response = str(record.get("response", "")).strip()
                if not instruction or not response:
                    instruction, response = parse_chat_text(text)
                domain = str(record.get("domain", "unknown")).strip() or "unknown"
                source = str(record.get("source", "unknown")).strip() or "unknown"
                examples.append(
                    ChatExample(
                        text=text,
                        instruction=instruction,
                        response=response,
                        domain=domain,
                        source=source,
                    )
                )
    if not examples:
        raise ValueError(f"No chat texts found in {path}")
    return examples


def limit_examples(examples: list[ChatExample], max_examples: int | None) -> list[ChatExample]:
    if max_examples is None:
        return examples
    if max_examples <= 0:
        raise ValueError("--max-examples must be positive when provided")
    return examples[:max_examples]


def split_examples(
    examples: list[ChatExample],
    val_ratio: float = 0.05,
) -> tuple[list[ChatExample], list[ChatExample]]:
    split_at = max(1, int(len(examples) * (1.0 - val_ratio)))
    return examples[:split_at], examples[split_at:] or examples[-1:]


def example_texts(examples: list[ChatExample]) -> list[str]:
    return [format_sft_text(example) for example in examples]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def has_technical_keyword(text: str) -> bool:
    lower = text.casefold()
    return any(re.search(rf"\b{re.escape(keyword)}s?\b", lower) for keyword in CURATED_TECHNICAL_KEYWORDS)


def is_overly_generic(example: ChatExample) -> bool:
    instruction = collapse_spaces(example.instruction).casefold()
    if any(instruction.startswith(prefix) for prefix in GENERIC_INSTRUCTION_PREFIXES):
        return True
    if example.domain == "general" and not has_technical_keyword(f"{example.instruction} {example.response}"):
        return True
    return False


def response_has_repeated_phrase(response: str) -> bool:
    return repeated_ngram(response, n=3) is not None or repeated_ngram(response, n=4) is not None


def is_curated_candidate(example: ChatExample) -> bool:
    instruction_words = word_count(example.instruction)
    response_words = word_count(example.response)
    if not CURATED_MIN_INSTRUCTION_WORDS <= instruction_words <= CURATED_MAX_INSTRUCTION_WORDS:
        return False
    if not CURATED_MIN_RESPONSE_WORDS <= response_words <= CURATED_MAX_RESPONSE_WORDS:
        return False
    if example.domain not in {"ai", "databases", "software_engineering", "cybersecurity", "general"}:
        return False
    if is_overly_generic(example):
        return False
    if response_has_repeated_phrase(example.response):
        return False
    return True


def curated_bucket(example: ChatExample) -> str | None:
    if example.domain in {"ai", "databases", "software_engineering"}:
        return example.domain
    if example.domain in {"cybersecurity", "general"}:
        return "cybersecurity_general"
    return None


def take_evenly(examples: list[ChatExample], count: int) -> list[ChatExample]:
    if count <= 0:
        return []
    if len(examples) <= count:
        return examples[:]
    if count == 1:
        return [examples[0]]
    step = (len(examples) - 1) / (count - 1)
    indices = [round(i * step) for i in range(count)]
    return [examples[index] for index in indices]


def target_bucket_counts(max_examples: int) -> dict[str, int]:
    base = max_examples // len(CURATED_BUCKETS)
    remainder = max_examples % len(CURATED_BUCKETS)
    return {
        bucket: base + (1 if idx < remainder else 0)
        for idx, bucket in enumerate(CURATED_BUCKETS)
    }


def build_curated_examples(
    examples: list[ChatExample],
    *,
    max_examples: int = CURATED_MAX_EXAMPLES,
) -> list[ChatExample]:
    max_examples = min(max_examples, CURATED_MAX_EXAMPLES)
    if max_examples <= 0:
        raise ValueError("Curated max examples must be positive")

    candidates = [example for example in examples if is_curated_candidate(example)]
    buckets: dict[str, list[ChatExample]] = {bucket: [] for bucket in CURATED_BUCKETS}
    for example in candidates:
        bucket = curated_bucket(example)
        if bucket is not None:
            buckets[bucket].append(example)

    selected: list[ChatExample] = []
    used_ids: set[int] = set()
    targets = target_bucket_counts(max_examples)
    for bucket, target in targets.items():
        bucket_examples = buckets[bucket]
        if bucket == "cybersecurity_general":
            cybersecurity = [example for example in bucket_examples if example.domain == "cybersecurity"]
            general = [example for example in bucket_examples if example.domain == "general"]
            chosen = take_evenly(cybersecurity, target)
            if len(chosen) < target:
                chosen.extend(take_evenly(general, target - len(chosen)))
        else:
            chosen = take_evenly(bucket_examples, target)

        for example in chosen:
            if id(example) not in used_ids:
                selected.append(example)
                used_ids.add(id(example))

    if not selected:
        raise ValueError("Curated mode found no examples after filtering")
    return selected[:max_examples]


def domain_counts(examples: list[ChatExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.domain] = counts.get(example.domain, 0) + 1
    return dict(sorted(counts.items()))


def weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
    loss = loss.view(weights.shape)
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


@torch.no_grad()
def estimate_loss(model: GPT, train_set: SFTDataset, val_set: SFTDataset, batch_size: int, device: torch.device, batches: int = 10) -> tuple[float, float]:
    model.eval()
    values: dict[str, float] = {}
    for name, dataset in [("train", train_set), ("val", val_set)]:
        losses = []
        for _ in range(batches):
            x, y, w = dataset.get_batch(batch_size, device)
            logits = model(x)
            losses.append(weighted_cross_entropy(logits, y, w).item())
        values[name] = sum(losses) / len(losses)
    model.train()
    return values["train"], values["val"]


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, step: int, cfg: GPTConfig, train_loss: float, val_loss: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg.__dict__,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "phase": "18B_marshmello_instruct",
    }
    torch.save(payload, path)
    return path.stat().st_size


def build_model(vocab_size: int, cfg: GPTConfig, device: torch.device) -> GPT:
    return GPT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(device)


def total_parameters(model: GPT) -> int:
    return sum(p.numel() for p in model.parameters())


def format_keys(keys: list[str] | tuple[str, ...], *, max_keys: int = 12) -> str:
    if not keys:
        return "none"
    shown = ", ".join(keys[:max_keys])
    if len(keys) > max_keys:
        shown += f", ... (+{len(keys) - max_keys} more)"
    return shown


def checkpoint_model_state(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model_state dict")
    return state  # type: ignore[return-value]


def verify_tokenizer_checkpoint_compatibility(
    *,
    bpe,
    model: GPT,
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    token_emb = checkpoint_state.get("token_emb.weight")
    if token_emb is None:
        raise KeyError("Checkpoint is missing token_emb.weight")
    if token_emb.ndim != 2:
        raise ValueError(f"Checkpoint token_emb.weight must be 2D, got shape {tuple(token_emb.shape)}")

    lm_head = checkpoint_state.get("lm_head.weight")
    checkpoint_vocab_size = int(token_emb.shape[0])
    print("Tokenizer/checkpoint compatibility:")
    print(f"  tokenizer vocab size:          {bpe.vocab_size:,}")
    print(f"  model vocab size:              {model.vocab_size:,}")
    print(f"  checkpoint token_emb.weight:   {tuple(token_emb.shape)}")
    if lm_head is not None:
        print(f"  checkpoint lm_head.weight:     {tuple(lm_head.shape)}")

    assert bpe.vocab_size == checkpoint_vocab_size, (
        f"Tokenizer vocab size {bpe.vocab_size} does not match checkpoint embedding "
        f"rows {checkpoint_vocab_size}"
    )
    assert model.vocab_size == checkpoint_vocab_size, (
        f"Model vocab size {model.vocab_size} does not match checkpoint embedding "
        f"rows {checkpoint_vocab_size}"
    )
    assert tuple(model.token_emb.weight.shape) == tuple(token_emb.shape), (
        f"Model token embedding shape {tuple(model.token_emb.weight.shape)} does not "
        f"match checkpoint shape {tuple(token_emb.shape)}"
    )
    if lm_head is not None:
        assert lm_head.shape[0] == checkpoint_vocab_size, (
            f"Checkpoint lm_head rows {lm_head.shape[0]} do not match token embedding "
            f"rows {checkpoint_vocab_size}"
        )
        assert tuple(model.lm_head.weight.shape) == tuple(lm_head.shape), (
            f"Model lm_head shape {tuple(model.lm_head.weight.shape)} does not match "
            f"checkpoint shape {tuple(lm_head.shape)}"
        )


def load_base_checkpoint_verified(
    path: Path,
    model: GPT,
    bpe,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint_model_state(checkpoint)
    verify_tokenizer_checkpoint_compatibility(
        bpe=bpe,
        model=model,
        checkpoint_state=checkpoint_state,
    )

    before_norm = float(model.token_emb.weight[0].detach().norm().item())
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    after_norm = float(model.token_emb.weight[0].detach().norm().item())

    print("Checkpoint load verification:")
    print(f"  missing keys:                 {format_keys(missing_keys)}")
    print(f"  unexpected keys:              {format_keys(unexpected_keys)}")
    print(f"  first embedding norm before:  {before_norm:.6f}")
    print(f"  first embedding norm after:   {after_norm:.6f}")
    print(f"  first embedding norm delta:   {after_norm - before_norm:+.6f}")

    missing_transformer_keys = [
        key for key in missing_keys if key.startswith(TRANSFORMER_KEY_PREFIXES)
    ]
    assert not missing_transformer_keys, (
        "Missing transformer keys after checkpoint load: "
        f"{format_keys(missing_transformer_keys)}"
    )
    return int(checkpoint.get("step", 0)) if isinstance(checkpoint, dict) else 0


def run_decode_sanity_check(bpe, examples: list[ChatExample], *, limit: int = 3) -> None:
    print("Decode sanity check:")
    for idx, example in enumerate(examples[:limit], start=1):
        formatted = format_sft_text(example)
        ids = encode_text(bpe, formatted)
        decoded = decode_ids_pretty(bpe, ids)
        print(f"  example {idx}:")
        print(f"    original: {one_line(formatted)}")
        print(f"    decoded:  {one_line(decoded)}")

        positions = [decoded.find(tag) for tag in (USER_TAG, ASSISTANT_TAG, END_TAG)]
        assert all(pos >= 0 for pos in positions), (
            f"Decoded example {idx} lost one of {USER_TAG}, {ASSISTANT_TAG}, {END_TAG}"
        )
        assert positions == sorted(positions), (
            f"Decoded example {idx} changed chat tag order: {positions}"
        )
        print(f"    tags:     {USER_TAG}=ok {ASSISTANT_TAG}=ok {END_TAG}=ok")


def set_trainable_parameters(model: GPT, *, freeze_backbone: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.blocks[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

    trainable = count_parameters(model)
    total = total_parameters(model)
    if trainable == 0:
        raise RuntimeError("No trainable parameters selected")
    if freeze_backbone:
        print("Freeze mode: trainable modules are lm_head + last transformer block")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")


def repeated_ngram(text: str, *, n: int = 3) -> tuple[str, ...] | None:
    words = re.findall(r"\S+", text.lower())
    if len(words) < n * 2:
        return None
    seen: set[tuple[str, ...]] = set()
    for start in range(0, len(words) - n + 1):
        ngram = tuple(words[start : start + n])
        if ngram in seen:
            return ngram
        seen.add(ngram)
    return None


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    *,
    penalty: float,
    window: int = 64,
) -> torch.Tensor:
    if penalty <= 1.0 or not token_ids:
        return logits
    adjusted = logits.clone()
    for token_id in set(token_ids[-window:]):
        if adjusted[token_id] > 0:
            adjusted[token_id] /= penalty
        else:
            adjusted[token_id] *= penalty
    return adjusted


@torch.no_grad()
def generate_assistant_reply(
    model: GPT,
    bpe,
    prompt: str,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> str:
    was_training = model.training
    model.eval()
    prefix = format_chat_prompt(prompt)
    ids = encode_text(bpe, prefix) or [0]
    end_ids = encode_text(bpe, END_TAG)
    prompt_len = len(ids)
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.block_size :]], dtype=torch.long, device=device)
        logits = model.generate_step_logits(context)[0]
        logits = apply_repetition_penalty(
            logits,
            ids[prompt_len:],
            penalty=repetition_penalty,
        )
        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = apply_top_k(logits / max(temperature, 1e-6), top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
        if end_ids and len(ids) >= len(end_ids) and ids[-len(end_ids) :] == end_ids:
            break

    if was_training:
        model.train()

    answer_ids = ids[prompt_len:]
    text = decode_ids_pretty(bpe, answer_ids)
    return text.split(END_TAG, 1)[0].strip()


def run_generation_eval(model: GPT, bpe, *, step: int) -> bool:
    print(f"  generation eval step {step}:")
    is_safe = True
    for prompt in GENERATION_EVAL_PROMPTS:
        reply = generate_assistant_reply(
            model,
            bpe,
            prompt,
            max_new_tokens=120,
            temperature=GENERATION_EVAL_TEMPERATURE,
            top_k=GENERATION_EVAL_TOP_K,
            repetition_penalty=GENERATION_EVAL_REPETITION_PENALTY,
        )
        print(f"    Q: {prompt}")
        print(f"    A: {one_line(reply)}")
        repeated = repeated_ngram(reply, n=3)
        if repeated is not None:
            print(f"  WARNING: repeated 3-gram detected: {' '.join(repeated)!r}", flush=True)
            is_safe = False
    return is_safe


def evaluate_exact_answer_match(
    model: GPT,
    bpe,
    examples: list[ChatExample],
) -> tuple[int, int]:
    matches = 0
    total = len(examples)
    for example in examples:
        expected_ids = encode_text(bpe, example.response)
        expected_text = decode_ids_pretty(bpe, expected_ids)
        max_new_tokens = max(32, min(len(expected_ids) + len(encode_text(bpe, END_TAG)) + 8, 256))
        generated = generate_assistant_reply(
            model,
            bpe,
            example.instruction,
            max_new_tokens=max_new_tokens,
        )
        if normalize_exact(generated) == normalize_exact(expected_text):
            matches += 1
    return matches, total


def train(args: argparse.Namespace) -> None:
    for interval_name in ("log_every", "eval_every", "generation_every", "checkpoint_every"):
        if getattr(args, interval_name) <= 0:
            raise ValueError(f"--{interval_name.replace('_', '-')} must be positive")

    learning_rate = args.lr
    if learning_rate is None:
        learning_rate = CURATED_DEFAULT_LR if args.mode == "curated" else SFT_DEFAULT_LR

    cfg = resolve_config(args.config)
    cfg = replace(
        cfg,
        max_steps=args.steps,
        learning_rate=learning_rate,
        grad_clip=SFT_GRAD_CLIP,
    )
    device = pick_device(force_cpu=args.cpu)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
    print(f"Loading tokenizer: {TOKENIZER_PATH}", flush=True)
    bpe = load_tokenizer(TOKENIZER_PATH)

    base_checkpoint = Path(args.base_checkpoint) if args.base_checkpoint else latest_checkpoint_for(cfg)
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_checkpoint}\n"
            "Run Phase 18A, retrain tokenizer, then pretrain base first:\n"
            "  python 18A_large_pretraining_corpus/build_corpus.py --target-words 1000000\n"
            "  python 13_gpt_pretraining/tokenizer/train_bpe.py\n"
            "  python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 3000"
        )

    print(f"Building model on {device}...", flush=True)
    model = build_model(bpe.vocab_size, cfg, device)
    print(f"Loading base checkpoint: {base_checkpoint}", flush=True)
    checkpoint_step = load_base_checkpoint_verified(base_checkpoint, model, bpe)

    print(f"Loading chat data: {args.data}", flush=True)
    examples = load_chat_examples(args.data)
    max_examples = args.max_examples
    if args.mode == "curated":
        curated_limit = max_examples or CURATED_MAX_EXAMPLES
        examples = build_curated_examples(examples, max_examples=curated_limit)
    elif args.mode == "overfit" and max_examples is None:
        max_examples = 20
        examples = limit_examples(examples, max_examples)
    else:
        examples = limit_examples(examples, max_examples)
    run_decode_sanity_check(bpe, examples)

    if args.mode == "overfit":
        train_examples = examples
        val_examples = examples
    else:
        train_examples, val_examples = split_examples(examples, val_ratio=0.05)

    print("Encoding SFT datasets...", flush=True)
    train_set = SFTDataset(example_texts(train_examples), bpe, cfg.block_size)
    val_set = SFTDataset(example_texts(val_examples), bpe, cfg.block_size)

    set_trainable_parameters(model, freeze_backbone=args.freeze_backbone)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=0.01)
    checkpoint_path = CURATED_LATEST_CHECKPOINT if args.mode == "curated" else LATEST_CHECKPOINT

    print("Phase 18B: Marshmello-45M-Instruct SFT")
    print("=" * 60)
    print(f"Config:          {cfg.config_name}")
    print(f"Mode:            {args.mode}")
    print(f"Device:          {device}")
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Base step:       {checkpoint_step}")
    print(f"Tokenizer:       {TOKENIZER_PATH}")
    print(f"Chat data:       {args.data}")
    print(f"Train examples:  {len(train_set):,}")
    print(f"Val examples:    {len(val_set):,}")
    if args.mode == "curated":
        print(f"Curated domains: {domain_counts(examples)}")
    print(f"Learning rate:   {cfg.learning_rate:.2e}")
    print(f"Grad clip norm:  {cfg.grad_clip:.1f}")
    print(f"Parameters:      {total_parameters(model):,}")
    print(f"Trainable:       {count_parameters(model):,}")
    print(f"Checkpoint path: {checkpoint_path}")
    print()

    model.train()
    t0 = time.perf_counter()
    last_train_loss = 0.0
    last_val_loss = 0.0
    baseline_exact_matches = 0
    best_exact_matches = 0
    best_val_loss = float("inf")
    last_generation_safe = True
    stop_reason: str | None = None

    if args.mode == "overfit" and not args.no_eval:
        baseline_exact_matches, exact_total = evaluate_exact_answer_match(model, bpe, train_examples)
        best_exact_matches = baseline_exact_matches
        print(
            f"Overfit exact answer match before training: "
            f"{baseline_exact_matches}/{exact_total}",
            flush=True,
        )

    for step in range(1, cfg.max_steps + 1):
        step_t0 = time.perf_counter()
        val_improved = False
        x, y, weights = train_set.get_batch(cfg.batch_size, device)
        logits = model(x)
        loss = weighted_cross_entropy(logits, y, weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        optimizer.step()

        tokens_sec = cfg.batch_size * cfg.block_size / max(time.perf_counter() - step_t0, 1e-6)
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} | train loss {loss.item():.4f} | {tokens_sec:,.0f} tok/s", flush=True)

        if not args.no_eval and (step % args.eval_every == 0 or step == cfg.max_steps):
            print("  running eval...", flush=True)
            last_train_loss, last_val_loss = estimate_loss(
                model, train_set, val_set, cfg.batch_size, device, batches=args.eval_batches
            )
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                val_improved = True
            elapsed = time.perf_counter() - t0
            avg_tps = step * cfg.batch_size * cfg.block_size / max(elapsed, 1e-6)
            print(
                f"  eval step {step} | train loss {last_train_loss:.4f} | "
                f"val loss {last_val_loss:.4f} | best val {best_val_loss:.4f} | "
                f"avg {avg_tps:,.0f} tok/s",
                flush=True,
            )

            if args.mode == "overfit":
                exact_matches, exact_total = evaluate_exact_answer_match(model, bpe, train_examples)
                print(
                    f"  overfit exact answer match {exact_matches}/{exact_total} "
                    f"(best {best_exact_matches}/{exact_total})",
                    flush=True,
                )
                if exact_matches > best_exact_matches:
                    best_exact_matches = exact_matches
                    print("  overfit exact answer match improved", flush=True)
                if exact_matches > baseline_exact_matches:
                    stop_reason = "overfit exact answer match improved"

        if not args.no_eval and step % args.generation_every == 0:
            last_generation_safe = run_generation_eval(model, bpe, step=step)
            if not last_generation_safe and args.eval_generation_only_warn:
                print(
                    "  generation warning only: checkpoint overwrite requires val improvement",
                    flush=True,
                )
            elif not last_generation_safe:
                stop_reason = "generated text became repetitive"

        if not args.no_save and (step % args.checkpoint_every == 0 or step == cfg.max_steps):
            if args.eval_generation_only_warn and not last_generation_safe and not val_improved:
                print(
                    "  skipping checkpoint: generation was repetitive and val did not improve",
                    flush=True,
                )
                continue
            print("  saving checkpoint...", flush=True)
            size = save_checkpoint(checkpoint_path, model, optimizer, step, cfg, last_train_loss, last_val_loss)
            print(f"  checkpoint path: {checkpoint_path} ({size / 1024**2:.1f} MB)", flush=True)

        if stop_reason is not None:
            print(f"WARNING: stopping early at step {step}: {stop_reason}", flush=True)
            break

    if stop_reason is None:
        print("\nTraining complete.")
    else:
        print(f"\nStopped early: {stop_reason}")
    print(f"Latest checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Marshmello-45M-Instruct.")
    parser.add_argument("--config", default="large_50m")
    parser.add_argument(
        "--mode",
        choices=("train", "overfit", "curated"),
        default="train",
        help=(
            "train: full SFT; overfit: use a tiny subset and track exact answer "
            "match; curated: filtered balanced small SFT set"
        ),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Path to base model checkpoint (default: latest for --config)",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--data", type=Path, default=CHAT_DATA_PATH)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-5 train/overfit, 2e-6 curated)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only lm_head and the final transformer block",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--generation-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument(
        "--eval-generation-only-warn",
        action="store_true",
        help=(
            "Treat repetitive generation as a warning; when warned, skip checkpoint "
            "overwrite unless validation loss improved."
        ),
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip eval, useful for one-step smoke tests")
    parser.add_argument("--no-save", action="store_true", help="Skip checkpoint save, useful for one-step smoke tests")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
