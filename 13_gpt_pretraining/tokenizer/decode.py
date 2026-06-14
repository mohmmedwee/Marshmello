"""BPE token decode with readable spacing for Phase 13 generation."""

from __future__ import annotations

import re

END_MARKER = "</w>"

# Small function words that often get glued when </w> is missing on prior token.
GLUED_WORDS: tuple[str, ...] = (
    "and",
    "or",
    "for",
    "the",
    "with",
    "to",
    "in",
    "on",
    "at",
    "by",
    "is",
    "are",
    "was",
    "from",
    "that",
    "this",
    "as",
    "an",
    "be",
    "of",
)


def decode_bpe_tokens(tokens: list[str], end_marker: str = END_MARKER) -> str:
    """
    Decode BPE token strings to text.

    Tokens ending with end_marker (default </w>) mark a completed word and
    receive a trailing space. Continuation subwords without end_marker are
    concatenated directly to the previous piece.
    """
    if not tokens:
        return ""

    pieces: list[str] = []
    for token in tokens:
        if token == "\n":
            if pieces and not pieces[-1].endswith("\n"):
                pieces.append("\n")
            else:
                pieces.append("\n")
            continue

        if token.endswith(end_marker):
            word = token[: -len(end_marker)]
            pieces.append(word)
            pieces.append(" ")
        else:
            pieces.append(token)

    return "".join(pieces).strip()


def clean_punctuation_spacing(text: str) -> str:
    """Remove space before punctuation; ensure space after when followed by a word."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([^\s.,;:!?])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_glued_words(text: str) -> str:
    """
    Insert missing spaces when </w> was absent on a hyphenated prefix (e.g. B-treeand).

    Only hyphenated compounds are split so correctly merged subwords like "This"
    (from ["Th", "is</w>"]) are not broken.
    """
    for word in GLUED_WORDS:
        text = re.sub(
            rf"([a-z0-9]+-[a-z0-9]+)({word})\b",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )
    return text


def fix_case_word_boundaries(text: str) -> str:
    """Insert space between lowercase/digit and following uppercase (camelCase gaps)."""
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)


def postprocess_text(text: str) -> str:
    """
    Final cleanup after BPE decode.

    - fix missing spaces from absent </w> markers
    - normalize punctuation spacing
    """
    text = fix_glued_words(text)
    text = fix_case_word_boundaries(text)
    text = clean_punctuation_spacing(text)
    return text.strip()


def decode_token_strings(tokens: list[str], end_marker: str = END_MARKER) -> str:
    """Decode BPE tokens and apply post-processing."""
    return postprocess_text(decode_bpe_tokens(tokens, end_marker=end_marker))


def decode_ids_pretty(bpe, ids: list[int]) -> str:  # noqa: ANN001 — BPETokenizer
    """Decode integer token IDs with Phase 13 spacing rules."""
    if not hasattr(bpe, "itos"):
        bpe.build_index()
    end_marker = getattr(bpe, "END", END_MARKER)
    tokens = [bpe.itos[i] for i in ids]
    return decode_token_strings(tokens, end_marker=end_marker)
