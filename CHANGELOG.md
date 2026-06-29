# Changelog

This changelog summarizes project milestones by phase. It is documentation-only and does not imply that every experimental checkpoint is a recommended release.

## Phase 19A

- Added `large_300m` scaling path for Marshmello-300M.
- Established a 268,834,816-parameter smoke/benchmark target.
- Documented throughput, memory, checkpoint size, and next training gates.

## Phase 18K

- Added a general assistant benchmark with 500 held-out examples.
- Split evaluation into five buckets: AI, databases, programming, system design, and general knowledge.
- Recorded the current 55M general benchmark result: 22.5% domain score and 64.2% hallucination for the best teacher-family checkpoint.
- Confirmed that broad SFT must be judged on 18K, while 18J remains a regression gate.

## Phase 18J

- Added the Marshmello core routing benchmark.
- Measured core concept routing on 100 held-out questions.
- Recorded the current best 55M routing result: 18%.

## Phase 18I

- Added routing-focused teacher experiments.
- Explored data changes intended to improve concept routing without broad assistant regression.

## Phase 18H

- Added chat-only adaptation experiments.
- Built chat-only corpus tooling for continued pretraining-style adaptation.

## Phase 18G

- Added checkpoint comparison tooling.
- Added expanded corpus construction for broader pretraining experiments.

## Phase 18E

- Added tiny teacher SFT datasets and evaluation.
- Built short, direct educational response data.
- Added math and transformer/database routing examples.

## Phase 18D

- Added tokenizer v2 experiments.
- Improved punctuation and chat marker coverage.
- Documented checkpoint incompatibility when changing token IDs.

## Phase 18C

- Added base chat adaptation with raw text plus chat boundary markers.
- Kept this stage as continued pretraining rather than masked SFT.

## Phase 18B

- Added Marshmello instruction tuning.
- Added chat CLI and SFT training modes.
- Added checkpoint handling for teacher, curated, and diagnostic overfit modes.

## Phase 18A

- Added a larger local technical pretraining corpus.
- Expanded coverage across AI/ML, databases, software engineering, cybersecurity, and Python/API-style text.

## Phase 17

- Added instruction dataset import and processing pipeline.
- Created JSONL-style instruction/response data preparation utilities.

## Phase 16

- Added evaluation suite for comparing small and larger checkpoints.
- Added memorization-oriented checks.

## Phase 15

- Added the 50M-class scaling configuration.
- Added gradient accumulation and MPS-oriented batch fallback.
- Added throughput and memory benchmarking for larger local training.

## Phase 14

- Added a local dataset pipeline with ingest, cleaning, dedupe, quality checks, sharding, and export.

## Phase 13

- Added decoder-only GPT pretraining.
- Added BPE tokenizer, causal attention, learned positional embeddings, checkpointing, resume, and generation CLI.

## Phases 01-12

- Built the educational foundation from one trainable weight through neural networks, embeddings, attention, transformers, language modeling, BPE tokenization, parameter scaling, and a simple instruction-tuning demo.
