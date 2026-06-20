# Phase 18I - Routing Teacher Fix

After Phase 18H + teacher SFT, Marshmello can produce short answers and stop at
`<END>`, but it **routes questions to the wrong definition**:

- `What is AI?` → a *tokenizer* answer
- `Who are you?` → generic assistant-ish text
- `Explain database indexes.` → works

The format and stopping are fine; the problem is **question-to-answer routing**.
Phase 18I fixes it with a tiny, high-signal SFT pass that maps many paraphrases
of each key concept to a consistent answer, while keeping the concepts' answer
vocabularies strictly disjoint.

## Why this works

- **Paraphrase density.** Each concept gets many phrasings of the same question
  (AI ×60, identity ×60, attention ×60, database indexes ×20, tokenizer/BPE ×70)
  so the model learns the *concept*, not one exact string. Tokenizer/BPE has the
  most examples because it was the concept that mis-routed to database indexes.
- **Consistent, not identical answers.** Each concept cycles through several
  answer variants that all say the same thing — no memorizing a single sentence.
  Answers are short, simple subject-verb sentences so the targets are clean
  (earlier output was on-topic but grammatically weak for AI, identity, and
  attention).
- **Hard negatives by construction.** The confusable concepts are trained side
  by side with **non-overlapping** answer vocabularies. The build fails if:
    - an AI/identity answer contains a tokenizer or database term,
    - a tokenizer answer contains a database / index / rows term,
    - a database answer contains a tokenizer / BPE / tokenization term.
  So the data can never teach the confusion we are removing.
- **Generic filler is banned.** Answers may not contain empty suffixes like
  *"It helps the system answer or act more usefully."*
- **First-answer-token weight = 30.** Training puts heavy loss weight on the
  very first answer token (assistant tokens only), which is exactly where
  routing is decided.

## Build data

```bash
python 18I_routing_teacher_fix/build_routing_data.py
```

Output: `18I_routing_teacher_fix/data/routing_teacher.jsonl` (270 examples).
Schema:

```json
{"instruction": "...", "response": "...", "domain": "...", "concept": "..."}
```

The build validates paraphrase counts, response length (6-45 words), the banned
generic phrases, and the hard-negative invariants, and fails loudly on any
violation.

## Train

First pass (from the teacher checkpoint):

```bash
python 18B_marshmello_instruct/train_instruct.py --mode routing
```

The `routing` mode (added to `train_instruct.py`) uses:

- data: `18I_routing_teacher_fix/data/routing_teacher.jsonl`
- base checkpoint (default): `18E_tiny_teacher_sft/checkpoints/teacher_latest.pt`
- first answer token weight: **30**, loss on assistant tokens only
- learning rate: **3e-6**
- steps: **300**

Output checkpoint: `18I_routing_teacher_fix/checkpoints/routing_latest.pt`.

**Patch pass** — after rebuilding the expanded dataset, continue training from
the existing routing checkpoint with a short, gentle run:

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --mode routing \
  --base-checkpoint 18I_routing_teacher_fix/checkpoints/routing_latest.pt \
  --steps 150 \
  --lr 1e-6
```

This resumes from and overwrites `routing_latest.pt`; back it up first if you
want to keep the pre-patch checkpoint.

Override any default via `--base-checkpoint`, `--lr`, `--steps`, `--data`.

## Evaluate

```bash
python 18I_routing_teacher_fix/eval_routing.py
```

Prompts: `What is AI?`, `Define artificial intelligence.`, `Who are you?`,
`What is attention?`, `Explain database indexes.`, `What is a tokenizer?`.

Each prompt is checked for:

- **on topic** — answer mentions at least one keyword for the right concept,
- **correctly routed** — answer contains none of the competing concept's
  signature terms. In particular, an **AI answer that contains `tokenizer`,
  `BPE`, `SQL`, or `database` fails**,
- **grammar** — fails on a repeated adjacent word (e.g. `answer answer`) or a
  malformed phrase where a concept noun is followed by a bare article with no
  verb (e.g. `Attention a model`),
- short (≤70 words) and no repeated 3-grams.

Use `--checkpoint PATH` to evaluate a specific checkpoint and `--no-strict` to
print results without a nonzero exit.
