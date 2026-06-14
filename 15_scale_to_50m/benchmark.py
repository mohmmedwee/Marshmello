#!/usr/bin/env python3
"""Benchmark the Phase 15 ~50M GPT config on Apple Silicon MPS."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "13_gpt_pretraining" / "benchmark.py"

if __name__ == "__main__":
    argv = [str(BENCHMARK), "--config", "large_50m", *sys.argv[1:]]
    sys.argv = argv
    runpy.run_path(str(BENCHMARK), run_name="__main__")
