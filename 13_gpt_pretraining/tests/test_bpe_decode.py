"""Tests for Phase 13 BPE decode spacing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from tokenizer.decode import (  # noqa: E402
    decode_bpe_tokens,
    decode_token_strings,
    postprocess_text,
)


class TestBPEDecode(unittest.TestCase):
    def test_split_subword_continuation(self) -> None:
        tokens = ["Th", "is</w>"]
        self.assertEqual(decode_bpe_tokens(tokens), "This")
        self.assertEqual(decode_token_strings(tokens), "This")

    def test_split_subword_multi_piece_word(self) -> None:
        tokens = ["data", "base</w>"]
        self.assertEqual(decode_token_strings(tokens), "database")

    def test_word_boundary_tokens(self) -> None:
        tokens = ["B-tree</w>", "and</w>"]
        self.assertEqual(decode_token_strings(tokens), "B-tree and")

    def test_punctuation_only_token(self) -> None:
        tokens = ["queries.</w>"]
        self.assertEqual(decode_token_strings(tokens), "queries.")

    def test_punctuation_in_word_token(self) -> None:
        tokens = ["SQL</w>", "for</w>", "queries.</w>"]
        self.assertEqual(decode_token_strings(tokens), "SQL for queries.")

    def test_missing_end_marker_on_prior_token(self) -> None:
        raw = decode_bpe_tokens(["B-tree", "and</w>"])
        self.assertEqual(raw, "B-treeand")
        self.assertEqual(postprocess_text(raw), "B-tree and")

    def test_no_space_before_punctuation(self) -> None:
        text = postprocess_text("Hello , world .")
        self.assertEqual(text, "Hello, world.")

    def test_space_after_punctuation(self) -> None:
        text = postprocess_text("Hello,world")
        self.assertEqual(text, "Hello, world")


if __name__ == "__main__":
    unittest.main()
