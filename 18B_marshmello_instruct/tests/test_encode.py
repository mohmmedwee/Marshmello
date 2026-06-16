"""Tests for safe BPE encoding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE13 = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from tokenizer.bpe_io import load_tokenizer  # noqa: E402
from tokenizer.encode import corpus_to_ids_safe  # noqa: E402
from config import TOKENIZER_PATH  # noqa: E402


class TestSafeEncode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not TOKENIZER_PATH.exists():
            raise unittest.SkipTest("tokenizer.json not built")
        cls.bpe = load_tokenizer(TOKENIZER_PATH)

    def test_question_mark_does_not_crash(self) -> None:
        text = "<USER> What is AI? <ASSISTANT> Artificial intelligence. <END>"
        ids = corpus_to_ids_safe(self.bpe, text)
        self.assertGreater(len(ids), 0)

    def test_unknown_unicode_stripped(self) -> None:
        text = "<USER> Explain π and emoji 🚀 <ASSISTANT> Math symbols. <END>"
        ids = corpus_to_ids_safe(self.bpe, text)
        self.assertGreater(len(ids), 5)


if __name__ == "__main__":
    unittest.main()
