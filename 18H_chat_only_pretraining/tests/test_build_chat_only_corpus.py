import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_chat_only_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_chat_only_corpus", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestInstructionFiltering(unittest.TestCase):
    def write_chat_data(
        self, records: list[tuple[str, str] | tuple[str, str, str]]
    ) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "chat.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                instruction, response = record[:2]
                domain = record[2] if len(record) == 3 else "general"
                text = f"<USER> {instruction} <ASSISTANT> {response} <END>"
                handle.write(json.dumps({"text": text, "domain": domain}) + "\n")
        return path

    def test_long_and_input_instructions_are_reported(self) -> None:
        response = "This answer contains enough simple words to pass the response filter."
        long_instruction = " ".join(["long"] * 81)
        input_instruction = "Input: " + " ".join(["value"] * 120)
        path = self.write_chat_data(
            [
                ("Explain a database index clearly.", response),
                (long_instruction, response),
                (input_instruction, response),
            ]
        )
        stats = MODULE.FilterStats()

        pairs = MODULE.load_instruction_examples(path, None, 80, stats, 42)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(stats.dropped_long_instructions, 2)
        self.assertEqual(stats.dropped_scaffolding_instruction, 1)
        self.assertEqual(stats.max_instruction_words_seen_input, 121)
        self.assertEqual(stats.max_instruction_words_seen_kept, 5)

    def test_input_rule_drops_short_context_dump(self) -> None:
        response = "This answer contains enough simple words to pass the response filter."
        input_instruction = "Explain this value. Input: four divided by sixteen."
        path = self.write_chat_data([(input_instruction, response)])
        stats = MODULE.FilterStats()

        pairs = MODULE.load_instruction_examples(path, None, 80, stats, 42)

        self.assertEqual(pairs, [])
        self.assertEqual(stats.dropped_long_instructions, 0)
        self.assertEqual(stats.dropped_scaffolding_instruction, 1)

    def test_response_containing_input_is_dropped(self) -> None:
        response = "Input: This answer otherwise contains enough words for the filter."
        path = self.write_chat_data([("Explain this value clearly.", response)])
        stats = MODULE.FilterStats()

        pairs = MODULE.load_instruction_examples(path, None, 80, stats, 42)

        self.assertEqual(pairs, [])
        self.assertEqual(stats.dropped_scaffolding_response, 1)

    def test_response_containing_instruction_is_dropped(self) -> None:
        response = "Instruction: This answer otherwise contains enough words for filtering."
        path = self.write_chat_data([("Explain this value clearly.", response)])
        stats = MODULE.FilterStats()

        pairs = MODULE.load_instruction_examples(path, None, 80, stats, 42)

        self.assertEqual(pairs, [])
        self.assertEqual(stats.dropped_scaffolding_response, 1)

    def test_source_records_are_shuffled_before_capping(self) -> None:
        response = "This answer contains enough simple words to pass the response filter."
        records = [
            (f"Question number {index} asks for a clear explanation.", response, domain)
            for index, domain in enumerate(("first", "second", "third", "fourth"))
        ]
        path = self.write_chat_data(records)
        stats = MODULE.FilterStats()

        pairs = MODULE.load_instruction_examples(path, 2, 80, stats, 42)

        self.assertEqual(
            [instruction for instruction, _ in pairs],
            [
                "Question number 2 asks for a clear explanation.",
                "Question number 1 asks for a clear explanation.",
            ],
        )
        self.assertEqual(stats.kept_instruction_domains, {"second", "third"})


class TestRepeatedBlockShuffle(unittest.TestCase):
    def block_arguments(self) -> dict[str, object]:
        return {
            "teacher_pairs": [
                ("Who are you?", "I am Marshmello, a small assistant built to answer questions."),
                ("What is AI?", "AI is software that learns patterns and performs useful tasks."),
            ],
            "instruction_pairs": [
                (
                    "Explain indexes.",
                    "Indexes help databases locate matching rows without scanning every row.",
                )
            ],
            "teacher_repeat": 4,
            "instruction_repeat": 3,
        }

    def test_repeated_blocks_are_shuffled_without_adjacent_duplicates(self) -> None:
        blocks = MODULE.build_blocks(**self.block_arguments(), seed=7)

        self.assertEqual(len(blocks), 11)
        self.assertTrue(all(left != right for left, right in zip(blocks, blocks[1:])))
        self.assertEqual(MODULE.max_consecutive_duplicate_blocks(blocks), 1)

    def test_seed_makes_shuffle_deterministic(self) -> None:
        first = MODULE.build_blocks(**self.block_arguments(), seed=42)
        second = MODULE.build_blocks(**self.block_arguments(), seed=42)

        self.assertEqual(first, second)

    def test_report_validation_rejects_low_teacher_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "teacher_ratio"):
            MODULE.validate_report(
                {
                    "max_consecutive_duplicate_blocks": 1,
                    "teacher_ratio_after_repeat": 0.29,
                }
            )

    def test_report_validation_rejects_adjacent_duplicate_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "consecutive duplicate blocks"):
            MODULE.validate_report(
                {
                    "max_consecutive_duplicate_blocks": 2,
                    "teacher_ratio_after_repeat": 0.60,
                }
            )


if __name__ == "__main__":
    unittest.main()
