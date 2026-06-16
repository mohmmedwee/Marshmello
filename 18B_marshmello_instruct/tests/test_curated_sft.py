"""Tests for curated SFT dataset selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "13_gpt_pretraining"))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from train_instruct import (  # noqa: E402
    ChatExample,
    build_curated_examples,
    domain_counts,
    format_chat_prompt,
    format_sft_text,
    is_curated_candidate,
)


def example(domain: str, instruction: str, response: str) -> ChatExample:
    return ChatExample(
        text=f"<USER>\n{instruction}\n<ASSISTANT>\n{response}\n<END>",
        instruction=instruction,
        response=response,
        domain=domain,
    )


def response(topic: str) -> str:
    return (
        f"{topic} systems use clear inputs, careful validation, measured changes, "
        "practical examples, focused tests, and simple explanations so a small "
        "model can learn useful behavior without memorizing broad unrelated text."
    )


class TestCuratedSFT(unittest.TestCase):
    def test_prompt_format_is_single_line_chat_prefix(self) -> None:
        self.assertEqual(
            format_chat_prompt("What is AI?"),
            "<USER> What is AI? <ASSISTANT>",
        )

    def test_sft_text_uses_exact_prompt_format(self) -> None:
        item = example("ai", "What is artificial intelligence?", response("Artificial intelligence"))
        self.assertTrue(format_sft_text(item).startswith("<USER> What is artificial intelligence? <ASSISTANT>"))
        self.assertTrue(format_sft_text(item).endswith(" <END>"))

    def test_curated_sampling_balances_domains(self) -> None:
        examples = [
            example("ai", "Explain artificial intelligence simply.", response("Artificial intelligence")),
            example("ai", "What does machine learning mean?", response("Machine learning")),
            example("databases", "Explain database indexes clearly.", response("Database index")),
            example("databases", "How do SQL queries use indexes?", response("SQL query")),
            example("software_engineering", "How should engineers test code?", response("Software engineering")),
            example("software_engineering", "Explain API design tradeoffs.", response("API design")),
            example("cybersecurity", "How does encryption protect data?", response("Cybersecurity encryption")),
            example("cybersecurity", "Explain network security alerts.", response("Network security")),
        ]

        selected = build_curated_examples(examples, max_examples=8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            domain_counts(selected),
            {
                "ai": 2,
                "cybersecurity": 2,
                "databases": 2,
                "software_engineering": 2,
            },
        )

    def test_curated_candidate_rejects_generic_instruction(self) -> None:
        item = example("general", "Give three tips for staying healthy.", response("General health"))
        self.assertFalse(is_curated_candidate(item))

    def test_curated_candidate_rejects_repeated_response_phrase(self) -> None:
        item = example(
            "ai",
            "Explain artificial intelligence clearly.",
            "computer is a computer and computer is a computer because models repeat phrases often",
        )
        self.assertFalse(is_curated_candidate(item))


if __name__ == "__main__":
    unittest.main()
