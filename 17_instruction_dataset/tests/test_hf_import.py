"""Tests for Phase 17B Hugging Face dataset import helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from import_hf_datasets import (  # noqa: E402
    ImportedPair,
    filter_and_dedupe,
    infer_domain,
    normalize_record,
)


class TestHFNormalization(unittest.TestCase):
    def test_normalizes_alpaca_schema(self) -> None:
        pair = normalize_record(
            {
                "instruction": "Explain machine learning.",
                "input": "Use simple terms.",
                "output": "Machine learning trains models from examples to make predictions.",
            },
            "tatsu-lab/alpaca",
        )
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertIn("Input:", pair.instruction)
        self.assertEqual(pair.domain, "ai")

    def test_normalizes_dolly_schema(self) -> None:
        pair = normalize_record(
            {
                "instruction": "Explain least privilege.",
                "context": "Security design",
                "response": "Least privilege grants only the access needed for a task.",
            },
            "databricks/databricks-dolly-15k",
        )
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertIn("Security design", pair.instruction)
        self.assertEqual(pair.domain, "cybersecurity")

    def test_codealpaca_defaults_to_software_engineering(self) -> None:
        pair = normalize_record(
            {
                "instruction": "Write a Python function.",
                "input": "",
                "output": "Define a function with def, pass inputs, and return a value.",
            },
            "sahil2801/CodeAlpaca-20k",
        )
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.domain, "software_engineering")


class TestHFFilters(unittest.TestCase):
    def test_filters_and_dedupes(self) -> None:
        pairs = [
            ImportedPair("Q1", "one two three four five six seven eight nine ten", "general", "a"),
            ImportedPair("Q1", "different response with enough words for filtering here", "general", "a"),
            ImportedPair("Q2", "one two three four five six seven eight nine ten", "general", "b"),
            ImportedPair("Q3", "too short", "general", "b"),
            ImportedPair(
                "Q4",
                " ".join(str(i) for i in range(12)),
                "software_engineering",
                "c",
            ),
            ImportedPair(
                "Q5",
                " ".join(str(i) for i in range(20)),
                "software_engineering",
                "c",
            ),
        ]
        kept, stats = filter_and_dedupe(
            pairs,
            min_response_words=5,
            max_response_words=15,
            max_examples=10,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(stats["duplicate_instructions_removed"], 1)
        self.assertEqual(stats["duplicate_responses_removed"], 1)
        self.assertEqual(stats["short_responses_removed"], 1)
        self.assertEqual(stats["long_responses_removed"], 1)
        self.assertEqual(stats["source_counts"], {"a": 1, "c": 1})

    def test_domain_inference(self) -> None:
        self.assertEqual(infer_domain("SQL query joins a table using an index", "x"), "databases")
        self.assertEqual(infer_domain("encrypt passwords and prevent phishing", "x"), "cybersecurity")
        self.assertEqual(infer_domain("write a python function with tests", "x"), "software_engineering")


if __name__ == "__main__":
    unittest.main()
