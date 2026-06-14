"""
Train a BPE tokenizer on data/corpus.txt and save tokenizer/tokenizer.json.

Reuses the educational BPETokenizer from phase 09.
Target vocabulary ≈ 8000 subword tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from bpe_demo import BPETokenizer  # noqa: E402

from config import DEFAULT_CONFIG, TOKENIZER_PATH  # noqa: E402
from data.prepare_corpus import prepare_corpus  # noqa: E402
from tokenizer.bpe_io import save_tokenizer  # noqa: E402


def train_bpe(
    corpus_path: Path | None = None,
    target_vocab_size: int | None = None,
    verbose: bool = True,
) -> BPETokenizer:
    target_vocab_size = target_vocab_size or DEFAULT_CONFIG.target_vocab_size
    corpus_path = corpus_path or prepare_corpus()
    text = corpus_path.read_text(encoding="utf-8")

    bpe = BPETokenizer()
    # First pass to learn initial char vocab size, then set merge count
    bpe.train(text, num_merges=0, verbose=False)
    initial = len(bpe.vocab)
    num_merges = max(0, target_vocab_size - initial)

    if verbose:
        print(f"Corpus: {len(text.split()):,} words")
        print(f"Initial BPE vocab: {initial}")
        print(f"Training {num_merges} merges → target ~{target_vocab_size} tokens")

    bpe = BPETokenizer()
    bpe.train(text, num_merges=num_merges, verbose=verbose)
    bpe.build_index()
    save_tokenizer(bpe, TOKENIZER_PATH)

    if verbose:
        print(f"Saved tokenizer → {TOKENIZER_PATH}")
        print(f"Final vocab size: {bpe.vocab_size:,}")

    return bpe


if __name__ == "__main__":
    train_bpe(verbose=True)
