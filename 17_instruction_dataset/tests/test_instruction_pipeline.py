"""Tests for Phase 17 instruction dataset pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from process_instructions import (  # noqa: E402
    DOMAINS,
    CHAT_PATH,
    INSTRUCTIONS_PATH,
    STATS_PATH,
    InstructionPair,
    clean_pairs,
    dataset_statistics,
    format_chat,
    run_pipeline,
)


class TestInstructionCleaning(unittest.TestCase):
    def test_dedupes_instruction_response_and_short_answer(self) -> None:
        records = [
            {
                "instruction": "Explain indexes.",
                "response": "Indexes help databases find matching rows without scanning every row.",
                "domain": "databases",
            },
            {
                "instruction": "Explain indexes.",
                "response": "Different response that should be removed by instruction dedupe.",
                "domain": "databases",
            },
            {
                "instruction": "What is an index?",
                "response": "Indexes help databases find matching rows without scanning every row.",
                "domain": "databases",
            },
            {"instruction": "Short?", "response": "Too short.", "domain": "general"},
        ]
        pairs, stats = clean_pairs(records, min_response_words=6)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(stats["duplicate_instructions_removed"], 1)
        self.assertEqual(stats["duplicate_responses_removed"], 1)
        self.assertEqual(stats["short_responses_removed"], 1)

    def test_rejects_unknown_domain(self) -> None:
        pairs, stats = clean_pairs(
            [
                {
                    "instruction": "Explain something.",
                    "response": "This response has enough words but the domain is invalid.",
                    "domain": "unknown",
                }
            ],
            min_response_words=6,
        )
        self.assertEqual(pairs, [])
        self.assertEqual(stats["invalid_records"], 1)


class TestExports(unittest.TestCase):
    def test_chat_format(self) -> None:
        pair = InstructionPair(
            instruction="Explain AI.",
            response="AI systems learn patterns from data to make useful predictions.",
            domain="ai",
        )
        self.assertEqual(
            format_chat(pair),
            (
                "<USER>\nExplain AI.\n<ASSISTANT>\n"
                "AI systems learn patterns from data to make useful predictions.\n<END>"
            ),
        )

    def test_pipeline_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "raw.jsonl"
            output_path = root / INSTRUCTIONS_PATH.name
            chat_path = root / CHAT_PATH.name
            stats_path = root / STATS_PATH.name
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "instruction": "Explain AI.",
                                "response": (
                                    "AI systems learn patterns from data and use those "
                                    "patterns to make predictions or decisions."
                                ),
                                "domain": "ai",
                            }
                        ),
                        json.dumps(
                            {
                                "instruction": "Explain least privilege.",
                                "response": (
                                    "Least privilege limits each account or service to "
                                    "only the access required for its job."
                                ),
                                "domain": "cybersecurity",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = run_pipeline(
                input_path=input_path,
                output_path=output_path,
                chat_path=chat_path,
                stats_path=stats_path,
                min_response_words=6,
            )
            self.assertEqual(report["total_pairs"], 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(chat_path.exists())
            self.assertTrue(stats_path.exists())

            row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(set(row), {"instruction", "response", "domain"})
            self.assertIn("<USER>", chat_path.read_text(encoding="utf-8"))


class TestStatistics(unittest.TestCase):
    def test_dataset_statistics(self) -> None:
        pairs = [
            InstructionPair("Q1", "one two three four five six", "ai", "alpaca"),
            InstructionPair("Q2", "one two three four five six seven eight", "general", "dolly"),
        ]
        stats = dataset_statistics(pairs)
        self.assertEqual(stats["total_pairs"], 2)
        self.assertEqual(stats["average_response_length_words"], 7.0)
        self.assertEqual(stats["domain_distribution"], {"ai": 1, "general": 1})
        self.assertEqual(stats["source_counts"], {"alpaca": 1, "dolly": 1})

    def test_required_domains(self) -> None:
        self.assertEqual(
            DOMAINS,
            {"software_engineering", "databases", "ai", "cybersecurity", "general"},
        )


if __name__ == "__main__":
    unittest.main()
