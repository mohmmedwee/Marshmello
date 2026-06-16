"""Tests for Phase 18C chat-format semantic evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from eval_chat_format import evaluate_generated_text  # noqa: E402


class TestEvalChatFormat(unittest.TestCase):
    def test_exact_start_ok_for_direct_ai_answer(self) -> None:
        result = evaluate_generated_text("AI is a field of computer science.", min_keywords=2)
        self.assertTrue(result.exact_start_ok)

    def test_semantic_ok_for_topic_keywords(self) -> None:
        result = evaluate_generated_text(
            "Machine learning uses data and algorithms to find patterns.",
            min_keywords=2,
        )
        self.assertTrue(result.semantic_ok)
        self.assertIn("machine", result.topic_keywords_found)
        self.assertIn("learning", result.topic_keywords_found)
        self.assertIn("data", result.topic_keywords_found)
        self.assertIn("algorithm", result.topic_keywords_found)

    def test_boundary_rejects_raw_corpus_start(self) -> None:
        result = evaluate_generated_text("teams should write this guidance clearly.", min_keywords=2)
        self.assertFalse(result.boundary_ok)


if __name__ == "__main__":
    unittest.main()
