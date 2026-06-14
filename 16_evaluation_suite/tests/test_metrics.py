"""Phase 16 evaluation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from metrics import (  # noqa: E402
    build_corpus_index,
    repeated_ngram_ratio,
    score_output,
)
from prompts import PROMPT_SUITE  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_repeated_ngram_detects_loops(self) -> None:
        text = "the cat sat on the mat the cat sat on the mat"
        ratio = repeated_ngram_ratio(text, n=4)
        self.assertGreater(ratio, 0.2)

    def test_exact_paragraph_match(self) -> None:
        corpus = build_corpus_index("Database systems store durable structured data.\n\nAI builds agents.")
        sample = score_output(
            prompt="Database systems",
            domain="databases",
            config_name="default",
            model_alias="Marshmello-8M",
            output="Database systems store durable structured data.",
            corpus=corpus,
        )
        self.assertTrue(sample.exact_paragraph_match)
        self.assertGreater(sample.nearest_similarity, 0.8)

    def test_prompt_suite_size(self) -> None:
        self.assertEqual(len(PROMPT_SUITE), 6)


if __name__ == "__main__":
    unittest.main()
