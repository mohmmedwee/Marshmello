"""
Phase 09: Educational Byte Pair Encoding (BPE) tokenizer

BPE is the subword method used by GPT-2, LLaMA, and many other LLMs.

Algorithm (simplified):
  1. Start with characters as the vocabulary.
  2. Count every pair of adjacent symbols in the training text.
  3. Merge the most frequent pair into one new symbol.
  4. Repeat step 2–3 for N merges.
  5. At encode time, apply the same merges to split text into subwords.

Why real LLMs use BPE:
  - Character-level: sequences are too long; hard to learn word meaning.
  - Word-level: huge vocab; any new word becomes <UNK>.
  - BPE: middle ground — common words stay whole, rare/new words split into pieces.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# Reuse the phase 08 corpus (same training text for fair comparison)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "08_word_level_language_model"))
from corpus import build_corpus  # noqa: E402


WORD_PATTERN = re.compile(r"\n|\w+|[^\w\s]", re.UNICODE)


def tokenize_char_level(text: str) -> list[str]:
    """Phase 07 style: one token per character (including spaces as implicit)."""
    return list(text)


def tokenize_word_level(text: str) -> list[str]:
    """Phase 08 style: words, punctuation, and newlines."""
    return WORD_PATTERN.findall(text)


def merge_pair(symbols: list[str], pair: tuple[str, str], replacement: str) -> list[str]:
    """Replace all non-overlapping occurrences of pair with replacement."""
    first, second = pair
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
            merged.append(replacement)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


class BPETokenizer:
    """
    Minimal byte-pair encoding tokenizer (educational, not production-grade).

    We train on whitespace-separated words. Each word starts as characters
    plus an end marker </w> so merges learn word boundaries.
    """

    END = "</w>"

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []
        self.vocab: set[str] = set()

    def _word_to_symbols(self, word: str) -> list[str]:
        if word == "\n":
            return ["\n"]
        return list(word) + [self.END]

    def train(self, text: str, num_merges: int, verbose: bool = False) -> None:
        """
        Learn merge rules from text.

        vocab before merges = unique characters (+ </w>, newline)
        vocab after merges  = initial vocab + one new token per merge
        """
        words = text.split()
        word_freq = Counter(words)

        # Each unique word → list of character symbols
        splits: dict[str, list[str]] = {
            word: self._word_to_symbols(word) for word in word_freq
        }

        self.vocab = set()
        for symbols in splits.values():
            self.vocab.update(symbols)

        initial_vocab_size = len(self.vocab)
        self.merges = []

        if verbose:
            print(f"  Initial vocab (characters + {self.END}): {initial_vocab_size}")

        for step in range(num_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, freq in word_freq.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab.add(new_token)

            for word in splits:
                splits[word] = merge_pair(splits[word], best_pair, new_token)

            if verbose and step < 8:
                print(f"  Merge {step + 1:4d}: {best_pair!r} → {new_token!r}")

        if verbose:
            print(f"  Final vocab after {len(self.merges)} merges: {len(self.vocab)}")
            print(f"  (+{len(self.vocab) - initial_vocab_size} subword tokens learned)")

        self._initial_vocab_size = initial_vocab_size

    @property
    def initial_vocab_size(self) -> int:
        return getattr(self, "_initial_vocab_size", len(self.vocab))

    def encode_word(self, word: str) -> list[str]:
        """Apply learned merges to one word (works on unseen words too)."""
        if word == "\n":
            return ["\n"]
        symbols = self._word_to_symbols(word)
        for pair in self.merges:
            symbols = merge_pair(symbols, pair, pair[0] + pair[1])
        return symbols

    def encode(self, text: str) -> list[str]:
        """Encode full text: split into words, BPE each word, flatten."""
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
        return tokens

    def decode(self, tokens: list[str]) -> str:
        """Rough decode: join tokens and replace </w> with space."""
        text = "".join(tokens)
        text = text.replace(self.END, " ")
        return text.strip()

    def build_index(self) -> None:
        """Assign integer IDs to every token in vocab (call after train)."""
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.itos: dict[int, str] = {i: tok for tok, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_ids(self, text: str) -> list[int]:
        """Encode text → list of integer token IDs."""
        if not hasattr(self, "stoi"):
            self.build_index()
        return [self.stoi[t] for t in self.encode(text)]

    def decode_ids(self, ids: list[int]) -> str:
        """Decode integer IDs → readable text."""
        if not hasattr(self, "itos"):
            self.build_index()
        return self.decode([self.itos[i] for i in ids])

    def corpus_to_ids(self, text: str) -> list[int]:
        """
        Tokenize a full multi-line corpus into BPE IDs.
        Splits on whitespace (same as training) so newlines separate words.
        """
        if not hasattr(self, "stoi"):
            self.build_index()
        ids: list[int] = []
        for word in text.split():
            for token_str in self.encode_word(word):
                ids.append(self.stoi[token_str])
        return ids


def word_level_with_unk(text: str, known_vocab: set[str]) -> list[str]:
    """
    Simulate phase 08 word tokenizer hitting an unknown word.
    Anything not in training vocab becomes <UNK>.
    """
    tokens = tokenize_word_level(text)
    return [t if t in known_vocab else "<UNK>" for t in tokens]


def print_token_row(label: str, tokens: list[str], max_tokens: int = 24) -> None:
    shown = tokens[:max_tokens]
    suffix = " ..." if len(tokens) > max_tokens else ""
    print(f"  {label:14s} ({len(tokens):3d} tokens): {shown}{suffix}")


def main() -> None:
    print("Phase 09: Byte Pair Encoding (BPE) tokenizer demo")
    print("=" * 60)

    corpus = build_corpus(min_words=10_000)
    print(f"Training corpus: {len(corpus.split()):,} words\n")

    # --- Train BPE ---
    num_merges = 800
    print(f"Training BPE with {num_merges} merges on phase 08 corpus...")
    bpe = BPETokenizer()
    bpe.train(corpus, num_merges=num_merges, verbose=True)

    print("\n--- Vocab size summary ---")
    char_vocab = len(set(corpus))  # unique characters in raw text
    word_vocab = len(set(tokenize_word_level(corpus)))
    print(f"  Character-level unique symbols: {char_vocab}")
    print(f"  Word-level unique tokens:         {word_vocab}")
    print(f"  BPE before merges:                {bpe.initial_vocab_size}")
    print(f"  BPE after {len(bpe.merges)} merges:           {len(bpe.vocab)}")

    # --- Side-by-side on a known sentence from the corpus ---
    sample = "To be, or not to be, that is the question"
    print(f"\n--- Same sentence, three tokenizations ---")
    print(f'Sentence: "{sample}"\n')

    char_tokens = tokenize_char_level(sample)
    word_tokens = tokenize_word_level(sample)
    bpe_tokens = bpe.encode(sample)

    print_token_row("Character", char_tokens)
    print_token_row("Word", word_tokens)
    print_token_row("BPE", bpe_tokens)

    print("\n  Notice: BPE uses fewer tokens than characters,")
    print("  but more than whole words when words split into subwords.")

    # --- Unknown / new words: BPE vs word-level ---
    print("\n--- Unknown words (NOT in training corpus) ---")
    print("Word-level tokenizer must use <UNK>. BPE splits into known subword pieces.\n")

    training_words = set(corpus.split())
    known_word_tokens = set(tokenize_word_level(corpus))

    new_sentences = [
        "ChatGPT tokenization is fascinating",
        "supercalifragilistic expialidocious",
        "unhappiness and remorsefulness",
    ]

    for sentence in new_sentences:
        print(f'Sentence: "{sentence}"')
        unk_tokens = word_level_with_unk(sentence, known_word_tokens)
        bpe_tokens = bpe.encode(sentence)

        has_unk = "<UNK>" in unk_tokens
        print(f"  Word-level: {unk_tokens}")
        print(f"  Has <UNK>:  {has_unk}")
        print(f"  BPE:        {bpe_tokens}")
        print(f"  BPE decode: \"{bpe.decode(bpe_tokens)}\"")
        print()

    # --- Show how a rare long word gets split ---
    rare = "supercalifragilistic"
    print(f'--- Subword split example: "{rare}" ---')
    pieces = bpe.encode_word(rare)
    print(f"  Characters ({len(rare)} tokens): {list(rare)}")
    print(f"  BPE pieces ({len(pieces)} tokens): {pieces}")
    print("  Even though the full word was never seen, pieces come from learned merges.")

    # --- First merges learned (most common patterns in Shakespeare corpus) ---
    print("\n--- First 12 merges learned (most frequent pairs in corpus) ---")
    for i, pair in enumerate(bpe.merges[:12], start=1):
        merged = pair[0] + pair[1]
        print(f"  {i:2d}. {pair!r} → {merged!r}")

    print("\n" + "=" * 60)
    print("Takeaway: BPE gives open vocabulary — no <UNK> for new words,")
    print("reasonable sequence length, and vocab size controlled by num_merges.")
    print("Real LLMs use this idea at scale (GPT-2: ~50k merges).")


if __name__ == "__main__":
    main()
