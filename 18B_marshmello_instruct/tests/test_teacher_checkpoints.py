"""Tests for teacher checkpoint persistence."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "13_gpt_pretraining"))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from train_instruct import load_checkpoint_metric, save_checkpoint  # noqa: E402


class TestTeacherCheckpoints(unittest.TestCase):
    def test_checkpoint_stores_teacher_score_and_replaces_atomically(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        cfg = SimpleNamespace(config_name="test")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_best_score.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                step=100,
                cfg=cfg,
                train_loss=0.5,
                val_loss=0.6,
                teacher_keyword_score=0.75,
            )

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["teacher_keyword_score"], 0.75)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_existing_best_metric_is_loaded_for_cross_run_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_best_score.pt"
            torch.save({"teacher_keyword_score": 0.8}, path)

            best_score = load_checkpoint_metric(path, "teacher_keyword_score", -1.0)

            self.assertEqual(best_score, 0.8)
            self.assertFalse(0.7 > best_score)

    def test_missing_metric_refuses_to_overwrite_unknown_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_best_score.pt"
            torch.save({"val_loss": 0.4}, path)

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                load_checkpoint_metric(path, "teacher_keyword_score", -1.0)


if __name__ == "__main__":
    unittest.main()
