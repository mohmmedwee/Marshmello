"""
Generate text from a trained Phase 13 GPT checkpoint.

BPE note: </w> marks end-of-word inside a token, NOT end-of-generation.
Generation stops on max_new_tokens, optional sentence end, tokenizer EOS, or stop-sequence.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from bpe_demo import BPETokenizer  # noqa: E402

from config import (  # noqa: E402
    DEFAULT_CONFIG,
    GPTConfig,
    latest_checkpoint_for,
    resolve_config,
)
from model.gpt import GPT  # noqa: E402
from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from tokenizer.decode import decode_ids_pretty, postprocess_text  # noqa: E402
from training.trainer import load_checkpoint, pick_device  # noqa: E402

DEFAULT_TEST_PROMPTS = [
    "To be",
    "The king",
    "Love is",
    "Artificial intelligence",
    "Database systems",
]

DEFAULT_STOP_SEQUENCE = "=== Topic:"
PARAGRAPH_PREFIX = "Write one coherent paragraph about "

EOS_TOKEN_NAMES = ("<EOS>", "<|endoftext|>", "<|end|>", "<END>")

DOMAIN_HINT_TEMPLATES: dict[str, str] = {
    "databases": "This text is about databases and data systems. ",
    "database": "This text is about databases and data systems. ",
    "machine_learning": "This text is about machine learning and model training. ",
    "artificial_intelligence": "This text is about artificial intelligence. ",
    "software_engineering": "This text is about software engineering and reliable systems. ",
    "deep_learning": "This text is about deep learning and neural networks. ",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class GenerateResult:
    text: str
    stop_reason: str
    new_tokens: int
    raw_text: str = ""
    matched_stop_sequence: str | None = None


def find_eos_token_id(bpe: BPETokenizer) -> int | None:
    if not hasattr(bpe, "stoi"):
        bpe.build_index()
    for name in EOS_TOKEN_NAMES:
        if name in bpe.stoi:
            return bpe.stoi[name]
    return None


def format_domain_hint(hint: str) -> str:
    key = hint.strip().lower().replace(" ", "_").replace("-", "_")
    if key in DOMAIN_HINT_TEMPLATES:
        return DOMAIN_HINT_TEMPLATES[key]
    readable = hint.strip().replace("_", " ")
    return f"This text is about {readable}. "


def format_prompt(
    prompt: str,
    *,
    style: str | None = None,
    domain_hint: str | None = None,
) -> str:
    parts: list[str] = []
    if domain_hint:
        parts.append(format_domain_hint(domain_hint))
    if style == "paragraph":
        parts.append(f"{PARAGRAPH_PREFIX}{prompt}")
    else:
        parts.append(prompt)
    return "".join(parts)


def encode_prompt(bpe: BPETokenizer, text: str) -> list[int]:
    """Encode text, skipping characters absent from the trained BPE vocab."""
    if not hasattr(bpe, "stoi"):
        bpe.build_index()

    known_chars = {tok for tok in bpe.vocab if len(tok) == 1}
    cleaned_chars: list[str] = []
    for ch in text:
        if ch.isspace():
            cleaned_chars.append(" ")
        elif ch in known_chars:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    cleaned = re.sub(r"\s+", " ", "".join(cleaned_chars)).strip()

    ids: list[int] = []
    for word in cleaned.split():
        for token_str in bpe.encode_word(word):
            if token_str in bpe.stoi:
                ids.append(bpe.stoi[token_str])
    return ids if ids else [0]


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return filtered


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, sorted_indices, sorted_logits)
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float = 1.15,
    window: int = 64,
) -> torch.Tensor:
    if penalty <= 1.0 or not token_ids:
        return logits
    adjusted = logits.clone()
    recent = token_ids[-window:]
    for token_id, count in Counter(recent).items():
        factor = penalty**count
        if adjusted[token_id] > 0:
            adjusted[token_id] /= factor
        else:
            adjusted[token_id] *= factor
    return adjusted


def apply_presence_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float = 0.0,
) -> torch.Tensor:
    if penalty <= 0.0 or not token_ids:
        return logits
    adjusted = logits.clone()
    for token_id in set(token_ids):
        adjusted[token_id] -= penalty
    return adjusted


def truncate_at_stop_sequence(text: str, stop_sequence: str | None) -> tuple[str, bool]:
    if not stop_sequence:
        return text, False
    idx = text.find(stop_sequence)
    if idx < 0:
        return text, False
    return text[:idx].rstrip(), True


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]


def ends_with_complete_sentence(text: str) -> bool:
    text = text.rstrip()
    return bool(text) and text[-1] in ".!?"


def generated_suffix(full_text: str, prompt_decoded: str) -> str:
    if full_text.startswith(prompt_decoded):
        return full_text[len(prompt_decoded) :]
    return full_text


def cleanup_sentences(
    text: str,
    *,
    min_sentences: int = 1,
    max_sentences: int = 5,
) -> str:
    text = text.rstrip()
    if not text:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return text

    if not ends_with_complete_sentence(text):
        if len(sentences) > 1:
            sentences = sentences[:-1]
        elif len(sentences) == 1 and not ends_with_complete_sentence(sentences[0]):
            return text.rstrip(" ,;:-")

    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    if len(sentences) < min_sentences:
        return text

    cleaned = " ".join(sentences)
    if not ends_with_complete_sentence(cleaned) and ends_with_complete_sentence(text):
        return text
    return cleaned


def postprocess_output(
    raw_text: str,
    *,
    stop_sequence: str | None,
    min_sentences: int,
    max_sentences: int,
) -> str:
    text, _ = truncate_at_stop_sequence(raw_text, stop_sequence)
    return cleanup_sentences(text, min_sentences=min_sentences, max_sentences=max_sentences)


def format_stop_reason(result: GenerateResult, bpe: BPETokenizer) -> str:
    if result.stop_reason == "stop_sequence" and result.matched_stop_sequence:
        return f"stop sequence matched ({result.matched_stop_sequence!r})"
    reasons = {
        "eos_token": "tokenizer EOS token",
        "sentence_end": "complete sentence after min_new_tokens",
        "max_tokens": f"max_new_tokens ({result.new_tokens} generated)",
        "invalid_logits": "invalid logits (numerical stop)",
        "word_boundary": (
            f"legacy BPE word boundary ({bpe.END!r}) — not a true EOS; avoid this flag"
        ),
    }
    return reasons.get(result.stop_reason, result.stop_reason)


@torch.no_grad()
def generate(
    model: GPT,
    bpe: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    *,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.15,
    presence_penalty: float = 0.6,
    repetition_window: int = 64,
    stop_sequence: str | None = DEFAULT_STOP_SEQUENCE,
    stop_on_eos_token: bool = True,
    stop_on_word_boundary: bool = False,
    stop_on_sentence_end: bool = False,
    min_new_tokens: int = 20,
    min_sentences: int = 1,
    max_sentences: int = 5,
    device: torch.device | None = None,
) -> GenerateResult:
    model.eval()
    device = device or next(model.parameters()).device

    ids = encode_prompt(bpe, prompt)
    prompt_len = len(ids)
    prompt_decoded = decode_ids_pretty(bpe, ids[:prompt_len])
    eos_id = find_eos_token_id(bpe)

    stop_reason = "max_tokens"
    generated = 0
    matched_stop: str | None = None

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.block_size :]], dtype=torch.long, device=device)
        logits = model.generate_step_logits(context)[0].clone()

        logits = apply_repetition_penalty(
            logits, ids, penalty=repetition_penalty, window=repetition_window
        )
        logits = apply_presence_penalty(logits, ids, penalty=presence_penalty)

        if greedy:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / max(temperature, 1e-6)
            logits = apply_top_k(logits, top_k)
            logits = apply_top_p(logits, top_p)
            if not torch.isfinite(logits).any():
                stop_reason = "invalid_logits"
                break
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if stop_on_eos_token and eos_id is not None and next_id == eos_id:
            stop_reason = "eos_token"
            break

        ids.append(next_id)
        generated += 1

        decoded = decode_ids_pretty(bpe, ids)
        suffix = generated_suffix(decoded, prompt_decoded)

        if stop_sequence and stop_sequence in suffix:
            stop_reason = "stop_sequence"
            matched_stop = stop_sequence
            break

        if (
            stop_on_sentence_end
            and generated >= min_new_tokens
            and ends_with_complete_sentence(suffix.strip())
            and len(split_sentences(suffix.strip())) >= 1
        ):
            stop_reason = "sentence_end"
            break

        if stop_on_word_boundary:
            token_str = bpe.itos[next_id]
            if token_str.endswith(bpe.END) and generated >= min_new_tokens:
                stop_reason = "word_boundary"
                break

    raw_text = decode_ids_pretty(bpe, ids)
    cleaned = postprocess_output(
        raw_text,
        stop_sequence=stop_sequence,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    cleaned = postprocess_text(cleaned)

    return GenerateResult(
        text=cleaned,
        raw_text=raw_text,
        stop_reason=stop_reason,
        new_tokens=generated,
        matched_stop_sequence=matched_stop,
    )


def build_mode_label(args: argparse.Namespace) -> str:
    if args.greedy:
        parts = ["greedy"]
    else:
        parts = [f"T={args.temperature}"]
        if args.top_k > 0:
            parts.append(f"top_k={args.top_k}")
        if args.top_p > 0:
            parts.append(f"top_p={args.top_p}")
    if args.repetition_penalty > 1.0:
        parts.append(f"rep={args.repetition_penalty}")
    if args.presence_penalty > 0:
        parts.append(f"presence={args.presence_penalty}")
    if args.style:
        parts.append(f"style={args.style}")
    if args.domain_hint:
        parts.append(f"hint={args.domain_hint}")
    if args.stop_on_sentence_end:
        parts.append("stop_on_sentence_end")
    return ", ".join(parts)


def print_generation(
    prompt: str,
    result: GenerateResult,
    bpe: BPETokenizer,
    mode: str,
    device: torch.device,
    *,
    show_stop: bool = True,
    show_raw: bool = False,
) -> None:
    print(f"Prompt:  {prompt!r}")
    print(f"Mode:    {mode}")
    print(f"Device:  {device}")
    print(f"Output:  {result.text}")
    if show_stop:
        print(f"Stopped: {format_stop_reason(result, bpe)}")
    if show_raw and result.raw_text != result.text:
        print(f"Raw:     {result.raw_text}")
    print()


def load_config_from_checkpoint(checkpoint: Path) -> GPTConfig | None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = ckpt.get("config")
    if not raw:
        return None
    return GPTConfig(**raw)


def load_model_and_tokenizer(
    checkpoint: Path,
    device: torch.device,
    cfg: GPTConfig,
) -> tuple[GPT, BPETokenizer, GPTConfig]:
    from config import TOKENIZER_PATH

    if not TOKENIZER_PATH.exists():
        print(f"Tokenizer not found: {TOKENIZER_PATH}")
        print("Run: python 13_gpt_pretraining/tokenizer/train_bpe.py")
        sys.exit(1)

    bpe = load_tokenizer(TOKENIZER_PATH)
    model_cfg = load_config_from_checkpoint(checkpoint) or cfg
    model = GPT(
        vocab_size=bpe.vocab_size,
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        num_layers=model_cfg.num_layers,
        d_ff=model_cfg.d_ff,
        block_size=model_cfg.block_size,
        dropout=0.0,
    ).to(device)
    load_checkpoint(checkpoint, model, optimizer=None, device=device)
    return model, bpe, model_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with Phase 13/15 GPT")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Model config: default | large_50m (selects checkpoint dir + architecture)",
    )
    parser.add_argument("--prompt", type=str, default="Artificial intelligence")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: latest for --config)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--min-new-tokens", type=int, default=20)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.0, help="0 disables nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--presence-penalty", type=float, default=0.6)
    parser.add_argument(
        "--stop-sequence",
        type=str,
        default=DEFAULT_STOP_SEQUENCE,
        help=f"Stop when generated text contains this string (default: {DEFAULT_STOP_SEQUENCE!r})",
    )
    parser.add_argument("--no-stop-sequence", action="store_true")
    parser.add_argument("--style", choices=["paragraph"], default=None)
    parser.add_argument(
        "--domain-hint",
        type=str,
        default=None,
        help='Prepend domain context, e.g. "databases"',
    )
    parser.add_argument("--min-sentences", type=int, default=1)
    parser.add_argument("--max-sentences", type=int, default=5)
    parser.add_argument(
        "--no-eos-stop",
        action="store_true",
        help="Do not stop on tokenizer EOS token (if vocab defines one)",
    )
    parser.add_argument(
        "--stop-on-word-boundary",
        action="store_true",
        help="Legacy: stop on BPE </w> marker (NOT recommended — </w> is not EOS)",
    )
    parser.add_argument(
        "--stop-on-sentence-end",
        action="store_true",
        help="Stop after first complete sentence once --min-new-tokens is reached",
    )
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--test-prompts", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint_for(cfg)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"Train first: python 13_gpt_pretraining/training/trainer.py --config {cfg.config_name}")
        sys.exit(1)

    device = pick_device()
    model, bpe, model_cfg = load_model_and_tokenizer(ckpt_path, device, cfg)
    mode = build_mode_label(args)
    stop_sequence = None if args.no_stop_sequence else args.stop_sequence

    prompts = DEFAULT_TEST_PROMPTS if args.test_prompts else [args.prompt]
    if args.test_prompts:
        print("--- Prompt test suite ---")
        print(f"Config: {model_cfg.config_name} | checkpoint: {ckpt_path.name}")
        print(f"Mode: {mode} | max_new_tokens={args.max_new_tokens}")
        print(f"Stop sequence: {stop_sequence!r}")
        print()
    elif model_cfg.config_name != "default":
        print(f"Config: {model_cfg.config_name} | checkpoint: {ckpt_path}")
        print()

    for user_prompt in prompts:
        model_prompt = format_prompt(
            user_prompt,
            style=args.style,
            domain_hint=args.domain_hint,
        )
        result = generate(
            model,
            bpe,
            model_prompt,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            stop_sequence=stop_sequence,
            stop_on_eos_token=not args.no_eos_stop,
            stop_on_word_boundary=args.stop_on_word_boundary,
            stop_on_sentence_end=args.stop_on_sentence_end,
            min_new_tokens=args.min_new_tokens,
            min_sentences=args.min_sentences,
            max_sentences=args.max_sentences,
            device=device,
        )
        print_generation(
            user_prompt,
            result,
            bpe,
            mode,
            device,
            show_raw=args.show_raw,
        )


if __name__ == "__main__":
    main()
