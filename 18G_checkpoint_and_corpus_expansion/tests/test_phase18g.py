"""Focused tests for Phase 18G metrics and synthetic corpus helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from build_expanded_corpus import DOMAIN_SPECS, synthetic_paragraph  # noqa: E402
from compare_checkpoints import (  # noqa: E402
    evaluate_text,
    repeated_ngram,
)


class TestCheckpointMetrics(unittest.TestCase):
    def test_repeated_three_gram(self) -> None:
        self.assertEqual(
            repeated_ngram("computer is a computer is a computer", n=3),
            ("computer", "is", "a"),
        )

    def test_topic_metrics_for_database_answer(self) -> None:
        result = evaluate_text(
            "<USER> Explain database indexes. <ASSISTANT>",
            "A database index is a lookup structure that finds rows faster for queries.",
        )
        self.assertGreater(result.topic_keyword_score, 0.4)
        self.assertFalse(result.repeated_3gram)
        self.assertTrue(result.chat_format_response_ok)

    def test_chat_boundary_rejects_new_user_tag(self) -> None:
        result = evaluate_text(
            "<USER> What is AI? <ASSISTANT>",
            "<USER> Artificial intelligence is a computer system.",
        )
        self.assertFalse(result.chat_format_response_ok)


class TestSyntheticCorpus(unittest.TestCase):
    def test_synthetic_paragraph_is_clean_and_substantial(self) -> None:
        topics = DOMAIN_SPECS["databases"]
        paragraph = synthetic_paragraph(
            "databases",
            topics[0],
            topics[1],
            "a production application",
            "keep operations predictable",
            0,
        )
        self.assertGreaterEqual(len(paragraph.split()), 45)
        self.assertIsNone(repeated_ngram(paragraph, n=3))


if __name__ == "__main__":
    unittest.main()
