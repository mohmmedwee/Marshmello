"""Tests for Phase 14 dataset pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))

from scripts.clean import clean_text, run_clean  # noqa: E402
from scripts.common import Document, hash_text, infer_domain, write_jsonl  # noqa: E402
from scripts.dedupe import run_dedupe  # noqa: E402
from scripts.ingest import html_to_text, ingest_file  # noqa: E402
from scripts.quality import run_quality, score_document  # noqa: E402
from scripts.shard import run_shard  # noqa: E402


class TestIngest(unittest.TestCase):
    def test_html_strips_navigation(self) -> None:
        html = "<nav>Home | Contact | Login</nav><p>Database indexes speed up queries.</p>"
        text = clean_text(html_to_text(html))
        self.assertNotIn("Home | Contact", text)
        self.assertIn("Database indexes", text)

    def test_ingest_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / "docs").mkdir()
            path = raw / "docs" / "note.md"
            path.write_text(
                "Machine learning models learn patterns from labeled training data.\n"
                "Evaluation metrics track accuracy, precision, recall, and calibration.\n",
                encoding="utf-8",
            )
            docs = ingest_file(path, raw)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].source, "docs")


class TestClean(unittest.TestCase):
    def test_remove_nav_line(self) -> None:
        raw = "Home | Contact | Login\n\nSQL joins combine rows from multiple tables."
        cleaned = clean_text(raw)
        self.assertNotIn("Home | Contact", cleaned)
        self.assertIn("SQL joins", cleaned)

    def test_normalize_whitespace(self) -> None:
        cleaned = clean_text("Too    many     spaces.\n\n\n\nSecond paragraph.")
        self.assertIn("Too many spaces.", cleaned)
        self.assertIn("Second paragraph.", cleaned)


class TestDedupe(unittest.TestCase):
    def test_exact_hash_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "in.jsonl"
            output_path = root / "out.jsonl"
            records = [
                {"source": "docs", "text": "Same paragraph about indexes and tables.", "language": "en"},
                {"source": "docs", "text": "Same paragraph about indexes and tables.", "language": "en"},
                {"source": "docs", "text": "Different paragraph about neural networks and training.", "language": "en"},
            ]
            write_jsonl(input_path, records)
            stats = run_dedupe(input_path, output_path)
            self.assertEqual(stats["duplicates_removed"], 1)
            self.assertEqual(stats["output_documents"], 2)


class TestQuality(unittest.TestCase):
    def test_filters_lorem_and_spam(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "in.jsonl"
            output_path = root / "out.jsonl"
            good = (
                "Software engineering teams design maintainable services with tests, "
                "code review, and observability for production reliability."
            )
            write_jsonl(
                input_path,
                [
                    {"source": "docs", "text": good, "language": "en"},
                    {"source": "docs", "text": "lorem ipsum dolor sit amet", "language": "en"},
                    {"source": "docs", "text": "aaaaaaaaaaaaaaaa", "language": "en"},
                ],
            )
            stats = run_quality(input_path, output_path, min_words=8)
            self.assertEqual(stats["output_documents"], 1)
            kept = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertIn("domain", kept)

    def test_domain_tagging(self) -> None:
        text = (
            "SQL queries join tables, filter rows, and aggregate metrics using indexes. "
            "Transactions guarantee consistency across concurrent database operations."
        )
        self.assertEqual(infer_domain(text), "databases")
        scores = score_document(text)
        self.assertGreater(int(scores["word_count"]), 10)


class TestShard(unittest.TestCase):
    def test_shard_byte_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "in.jsonl"
            output_dir = root / "shards"
            records = [
                {
                    "source": "docs",
                    "text": f"Document {idx} with enough words to consume shard bytes quickly.",
                    "language": "en",
                    "domain": "general",
                }
                for idx in range(20)
            ]
            write_jsonl(input_path, records)
            stats = run_shard(input_path, output_dir, max_bytes=300)
            self.assertGreaterEqual(stats["shards"], 2)


class TestHash(unittest.TestCase):
    def test_hash_is_stable_after_whitespace_normalize(self) -> None:
        left = "Same   text\n\nwith spacing"
        right = "Same text\n\nwith spacing"
        self.assertEqual(hash_text(left), hash_text(right))


if __name__ == "__main__":
    unittest.main()
