"""Safe BPE encoding when the corpus or chat text contains out-of-vocab characters."""

from __future__ import annotations

import re

from bpe_demo import BPETokenizer  # noqa: F401 — re-export type for callers


def known_single_chars(bpe: BPETokenizer) -> set[str]:
    return {tok for tok in bpe.vocab if len(tok) == 1}


def sanitize_word(word: str, known: set[str]) -> str:
    if word == "\n":
        return word
    return "".join(ch for ch in word if ch in known)


def corpus_to_ids_safe(bpe: BPETokenizer, text: str) -> list[int]:
    """
    Encode text to BPE IDs, dropping characters absent from the trained vocab.

    Unknown punctuation (e.g. ``?`` when not in BPE merges) is removed per word
    instead of raising KeyError. Whitespace splitting matches ``corpus_to_ids``.
    """
    if not hasattr(bpe, "stoi"):
        bpe.build_index()

    known = known_single_chars(bpe)
    ids: list[int] = []
    for word in text.split():
        safe_word = sanitize_word(word, known)
        if not safe_word:
            continue
        for token_str in bpe.encode_word(safe_word):
            token_id = bpe.stoi.get(token_str)
            if token_id is not None:
                ids.append(token_id)
    return ids


def encode_prompt_safe(bpe: BPETokenizer, text: str) -> list[int]:
    """Like ``generate.encode_prompt`` — normalize whitespace then encode safely."""
    if not hasattr(bpe, "stoi"):
        bpe.build_index()

    known = known_single_chars(bpe)
    cleaned_chars: list[str] = []
    for ch in text:
        if ch.isspace():
            cleaned_chars.append(" ")
        elif ch in known:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    cleaned = re.sub(r"\s+", " ", "".join(cleaned_chars)).strip()

    ids = corpus_to_ids_safe(bpe, cleaned)
    return ids if ids else [0]
