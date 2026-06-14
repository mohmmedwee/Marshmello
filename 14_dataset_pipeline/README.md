# Phase 14 — Dataset Pipeline

Phase 13 trained GPT from a single `corpus.txt`. Real LLM teams never do that. They build a **dataset pipeline** first — then train 10M, 50M, or 100M parameter models from the **same** cleaned, deduped, sharded data.

---

## Why data quality beats model size

A 100M parameter model trained on duplicated navigation menus, spam, and lorem ipsum will **memorize garbage** faster than it learns language. Companies like OpenAI, Meta, and Google spend more engineering time on **data** than on architecture tweaks.

| Bad data symptom | What the model learns |
|------------------|------------------------|
| Same README × 100 | Verbatim repetition |
| `Home \| Contact \| Login` | Navbar tokens |
| `aaaaaaaa` spam | Character loops |
| Mixed languages | Unstable generation |

**Better data at 10M params often beats messy data at 100M params.**

---

## Pipeline

```text
raw/          txt, md, jsonl, html, pdf
   ↓ ingest
ingested/     normalized JSONL
   ↓ clean
cleaned/      no nav lines, normalized whitespace
   ↓ dedupe
deduped/      SHA-256 exact dedupe
   ↓ quality
deduped/      domain tags + spam filter
   ↓ shard
shards/       shard_000.jsonl, shard_001.jsonl, …
   ↓ stats
reports/      dataset_report.json, domain_distribution.json
```

---

## Folder layout

```text
14_dataset_pipeline/
├── raw/
│   ├── books/
│   ├── docs/
│   ├── code/
│   └── wikipedia/
├── cleaned/
├── deduped/
├── shards/
├── reports/
├── ingested/
├── scripts/
│   ├── ingest.py
│   ├── clean.py
│   ├── dedupe.py
│   ├── quality.py
│   ├── shard.py
│   └── stats.py
├── tests/
├── run_pipeline.py
├── export_corpus.py
└── README.md
```

---

## Document schema

Every record is normalized to:

```json
{
  "source": "wikipedia",
  "domain": "ai",
  "text": "...",
  "language": "en",
  "meta": {
    "quality": {
      "word_count": 120,
      "unique_word_ratio": 0.62,
      "line_repetition_ratio": 0.0,
      "language_confidence": 0.91
    }
  }
}
```

**Domains:** `ai`, `databases`, `software_engineering`, `general`

---

## Quick start

From `mini-transformer-from-scratch/` with venv active:

```bash
# Run full pipeline on sample raw data
python 14_dataset_pipeline/run_pipeline.py

# Production shard size (100 MB)
python 14_dataset_pipeline/run_pipeline.py --shard-mb 100

# Export shards → Phase 13 corpus.txt
python 14_dataset_pipeline/export_corpus.py

# Train GPT (Phase 13)
python 13_gpt_pretraining/tokenizer/train_bpe.py
python 13_gpt_pretraining/training/trainer.py --quick
```

---

## Individual steps

```bash
cd 14_dataset_pipeline

python scripts/ingest.py
python scripts/clean.py --input ingested/documents.jsonl
python scripts/dedupe.py --input cleaned/documents.jsonl
python scripts/quality.py --input deduped/documents.jsonl --output deduped/filtered.jsonl
python scripts/shard.py --input deduped/filtered.jsonl --max-mb 100
python scripts/stats.py --shards-dir shards
```

---

## What each step does

### 1. Ingest (`ingest.py`)

Reads `txt`, `md`, `jsonl`, `html`, and `pdf` (requires optional `pypdf`) from `raw/`. Subfolder name becomes `source` (`wikipedia`, `docs`, `code`, …).

### 2. Clean (`clean.py`)

Removes:

- empty lines
- duplicate whitespace
- navigation lines (`Home | Contact | Login`)
- separator lines (`===`, `---`)
- footer boilerplate (`Copyright`, `All rights reserved`)

### 3. Dedupe (`dedupe.py`)

Exact deduplication with `sha256(normalized_text)`. Reports duplicate ratio.

### 4. Quality (`quality.py`)

Scores each document:

| Metric | Purpose |
|--------|---------|
| word count | drop tiny fragments |
| unique word ratio | drop repetitive spam |
| line repetition ratio | drop copy-pasted blocks |
| language confidence | drop non-English noise |

Filters `lorem ipsum`, character spam (`aaaaaaaa`), and low-quality text. Assigns `domain` via keyword scoring.

### 5. Shard (`shard.py`)

Splits into `shard_000.jsonl`, `shard_001.jsonl`, … at a configurable byte limit (default **100 MB** in production; **1 MB** in demo `run_pipeline.py`).

### 6. Statistics (`stats.py`)

Writes:

- `reports/dataset_report.json` — documents, words, estimated BPE tokens, duplicates removed
- `reports/domain_distribution.json` — domain percentages

Example console output:

```text
Documents: 6
Words: 312
BPE Tokens (est.): 421

Domains:
  software_engineering: 33.3%
  ai: 33.3%
  databases: 33.3%

Duplicates removed: 14.3%
```

---

## Tests

```bash
python -m unittest 14_dataset_pipeline.tests.test_pipeline -v
```

---

## Connect to Phase 13

```text
Phase 14 shards  →  export_corpus.py  →  corpus.txt  →  Phase 13 BPE + GPT
```

Same pipeline scales to larger raw dumps — swap `raw/` contents, increase `--shard-mb`, re-run.

---

## Success criteria

```text
✓ Multi-source ingestion (txt, md, jsonl, html, pdf*)
✓ Cleaning
✓ Deduplication (SHA-256)
✓ Quality filtering
✓ Domain tagging
✓ Sharding
✓ Dataset reports
✓ Ready for GPT training (export_corpus.py)
```

\* PDF extraction requires `pip install pypdf` (optional).
