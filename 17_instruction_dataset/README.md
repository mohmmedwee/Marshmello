# Phase 17 — Instruction Dataset Pipeline

Phase 17 prepares instruction tuning data for **Marshmello-45M-Instruct**.

Phase 14 built a raw-text pretraining pipeline. This phase builds a supervised
fine-tuning dataset: each row is a user instruction paired with an assistant
response and a domain label.

---

## Pretraining vs Instruction Tuning

**Pretraining** teaches a model general language patterns by predicting the next
token in raw text:

```text
Database indexes speed up queries by...
```

The model learns grammar, facts, code patterns, and broad world structure, but
it is not specifically trained to follow a user request.

**Instruction tuning** trains on formatted user-assistant examples:

```text
<USER>
Explain database indexes.
<ASSISTANT>
A database index is an auxiliary data structure...
<END>
```

This teaches the pretrained model how to route a prompt to a helpful answer,
when to answer as the assistant, and when to stop.

---

## Schema

Processed instruction rows are JSONL:

```json
{"instruction":"Explain database indexes.","response":"A database index...","domain":"databases"}
```

Required domains:

- `software_engineering`
- `databases`
- `ai`
- `cybersecurity`
- `general`

---

## Pipeline

```text
raw/seed_instructions.jsonl or raw/hf_imported.jsonl
   ↓ validate schema + domain
   ↓ clean whitespace
   ↓ dedupe instructions
   ↓ dedupe responses
   ↓ remove short answers
processed/instructions.jsonl
processed/chat.jsonl
reports/instruction_stats.json
```

`processed/instructions.jsonl` keeps the canonical schema:

```json
{"instruction":"...","response":"...","domain":"..."}
```

`processed/chat.jsonl` is the model-ready SFT export:

```text
<USER>
instruction
<ASSISTANT>
response
<END>
```

---

## Run

From the repo root:

```bash
python 17_instruction_dataset/process_instructions.py
```

If `raw/hf_imported.jsonl` exists, `process_instructions.py` uses it by
default. Otherwise it falls back to the small seed file.

---

## Phase 17B: Online Import

Install the Hugging Face dataset loader:

```bash
pip install datasets
```

Import public instruction datasets:

```bash
python 17_instruction_dataset/import_hf_datasets.py --max-examples 50000
```

Supported sources:

- `tatsu-lab/alpaca`
- `databricks/databricks-dolly-15k`
- `sahil2801/CodeAlpaca-20k`

The importer normalizes every source to:

```json
{"instruction":"...","response":"...","domain":"general","source":"tatsu-lab/alpaca"}
```

Then run the existing processor:

```bash
python 17_instruction_dataset/process_instructions.py
```

The final training schema remains:

```json
{"instruction":"...","response":"...","domain":"..."}
```

Useful options:

```bash
python 17_instruction_dataset/process_instructions.py \
  --input 17_instruction_dataset/raw/seed_instructions.jsonl \
  --output 17_instruction_dataset/processed/instructions.jsonl \
  --chat-output 17_instruction_dataset/processed/chat.jsonl \
  --stats-output 17_instruction_dataset/reports/instruction_stats.json \
  --min-response-words 6
```

---

## Statistics

The pipeline prints and writes:

- total pairs
- average response length
- domain distribution
- source counts
- cleaning counts for invalid rows, short answers, duplicate instructions, and
  duplicate responses

---

## Tests

```bash
python -m unittest 17_instruction_dataset.tests.test_instruction_pipeline -v
```

---

## Output

Required output:

```text
17_instruction_dataset/processed/instructions.jsonl
```

Additional SFT export:

```text
17_instruction_dataset/processed/chat.jsonl
```
