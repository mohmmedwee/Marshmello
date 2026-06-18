# Phase 18H - Chat-Only Format Pretraining

After the expanded mixed pretraining (Phase 18G), base v2 reaches a good
validation loss but still fails chat prompts: it produces weak answers and then
keeps emitting raw corpus text after `<END>`.

Phase 18H adds a short **chat-only** continued-pretraining stage that contains
nothing but chat turns, so the model strongly learns the boundary format:

```text
<USER> instruction <ASSISTANT> answer <END>
```

This is not SFT. The corpus is plain text used for next-token prediction on
every token, with no assistant masking. Teacher examples are repeated heavily so
the small, direct-answer behavior dominates the signal.

## Build Corpus

```bash
python 18H_chat_only_pretraining/build_chat_only_corpus.py
```

Inputs:

- `18E_tiny_teacher_sft/data/teacher.jsonl` — repeated 20x
- `17_instruction_dataset/processed/chat.jsonl` — capped at 5000 and repeated 1x

Only short responses (8-80 words) and instructions of at most 80 words are
kept. Examples are dropped if either side contains `Input:`, `Output:`,
`Instruction:`, or `Response:`. Instruction records are shuffled
deterministically before the 5000-example cap, avoiding a source-order bias.
Each example is written as a single line:

```text
<USER> instruction <ASSISTANT> response <END>
```

Output:

```text
13_gpt_pretraining/data/corpus_chat_only.txt
```

A JSON report is written to:

```text
18H_chat_only_pretraining/reports/chat_only_corpus_report.json
```

It records source and kept counts, total words, teacher ratio after repetition,
instruction filtering counts, longest input and kept instructions, and
duplicate block metrics. Examples are shuffled after repetition with no
consecutive duplicate blocks. The build fails if the teacher ratio is below
30% or any duplicate block remains adjacent.

Useful flags:

- `--teacher-repeat N` (default `20`)
- `--instruction-repeat N` (default `1`)
- `--max-instruction-words N` (default `80`)
- `--max-instruction-examples N` (default `5000`)
- `--seed N` (default `42`)

## Train

```bash
python 13_gpt_pretraining/training/trainer.py \
  --config large_50m \
  --resume 13_gpt_pretraining/checkpoints/large_50m/step_006500.pt \
  --corpus 13_gpt_pretraining/data/corpus_chat_only.txt \
  --steps 1000 \
  --lr 2e-5
```

A short run (1000 steps) at a low learning rate (`2e-5`) is enough: the goal is
to lock in the chat boundary format without erasing the knowledge learned during
mixed pretraining.

## Evaluate

```bash
python 18H_chat_only_pretraining/eval_chat_only.py
```

Prompts:

```text
<USER> What is AI? <ASSISTANT>
<USER> Who are you? <ASSISTANT>
<USER> Explain database indexes. <ASSISTANT>
```

Success criteria (per prompt):

- output before `<END>` is on topic
- output stops (or can be truncated) at `<END>`
- no raw corpus continuation after `<END>`
- no repeated 3-grams

The script prints `PASS`/`FAIL` per prompt and an overall result. Pass
`--no-strict` to print results without exiting nonzero, or `--checkpoint PATH`
to evaluate a specific checkpoint.
