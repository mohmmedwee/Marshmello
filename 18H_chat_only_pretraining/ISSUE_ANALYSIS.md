# Phase 18H - Why the chat output is wrong (diagnosis)

This document explains **exactly** why base v2 produces bad chat answers, based
on running the current checkpoint, not guesses. No code was changed to produce
this analysis.

## 1. What the model actually does right now

Running `eval_chat_only.py` against the current base
(`13_gpt_pretraining/checkpoints/large_50m/latest.pt`, = step ~7000):

| Prompt | Answer (verbatim) | Verdict |
|--------|-------------------|---------|
| `What is AI?` | "The main difference between a computer is that can be used to develop a branch of data and make predictions based on the data from the data and the data." | on topic by luck, but rambling |
| `Who are you?` | "The main difference between the two main difference between a and a group of the two different types of variables..." | off topic, loops `main difference between` |
| `Explain database indexes.` | "The concept refers to the process of a process involves identifiers... This can be tested on normal cases and failures so teams can make behavior easier to understand. To keep the result reliable, use simpler models..." | drifts into raw-corpus text, **never emits `<END>`** |

So the three concrete symptoms are:

1. **Generic, non-answering text.** It never directly answers the question.
2. **Degenerate loops** (`main difference between ...`).
3. **No stop.** On the database prompt it does not produce `<END>`; it keeps
   going with raw-corpus-style sentences. This is the "continues raw corpus
   after the answer" problem.

## 2. What is NOT the cause (ruled out with evidence)

These are the usual suspects, and they are all **fine** here:

- **Marker tokenization.** `<USER>`, `<ASSISTANT>`, and `<END>` each encode to a
  **single** token (`<USER></w>`, `<ASSISTANT></w>`, `<END></w>`). The model is
  not struggling to assemble the markers from sub-pieces.
- **Format mismatch.** The eval prompt uses single-line
  `<USER> ... <ASSISTANT>`. The corpus the base trained on
  (`corpus_chat_mixed_expanded.txt`) also stores chat turns on a **single line**
  in the same shape. So prompt format matches training format.
- **Decoding settings.** Eval uses greedy + `stop_sequence="<END>"`. The model
  simply does not assign `<END>` enough probability to stop — see below.

The problem is **the training data distribution**, not the plumbing.

## 3. Root cause: the assistant/`<END>` signal was drowned out

The base was pretrained on `corpus_chat_mixed_expanded.txt` (Phase 18G). From
its report:

- **`actual_chat_ratio` = 0.20** — only ~20% of tokens are chat turns; the other
  80% is raw paragraphs. So most of what the model learned is "continue prose,"
  which is exactly what it does after answering.
- Worse, **inside** those chat turns the answer is a tiny slice. The curated
  instruction data (`17_instruction_dataset/processed/chat.jsonl`) is mostly
  Alpaca, which puts a large `Input:` dump in the **USER** turn. Example from the
  corpus: a full ~2,000-word Nero Wikipedia passage in `<USER>`, then a 13-word
  `<ASSISTANT>` answer, then `<END>`.

Net effect on the three symptoms:

- **No stop / raw continuation (symptom 3).** The token `<END>` appears very
  rarely relative to ordinary running text, and almost always after a *long*
  answer. After the model emits a few sentences, its highest-probability next
  token is "more prose," not `<END>`. So it runs on.
- **Rambling, no direct answer (symptom 1).** Curated responses average ~58
  words and are multi-sentence (per `17_instruction_dataset/reports/instruction_stats.json`).
  The model learned "produce a long generic paragraph," not "answer crisply."
- **Identity failure on `Who are you?`.** The only identity/persona data is the
  250 teacher examples (`18E`), which are a rounding error inside a ~5M-word
  pretraining corpus, so the model has effectively never learned who it is.
- **Loops (symptom 2)** are ordinary small-model (≈45-50M) greedy degeneration,
  made worse because the model has no strong, specific answer to commit to.

**In one sentence:** the base never got a strong, repeated "short question →
short answer → `<END>`" signal, so it defaults to raw-corpus continuation.

## 4. How Phase 18H is meant to fix it

The 18H corpus (`corpus_chat_only.txt`) is **100% chat turns** (chat ratio goes
from 0.20 → 1.00), filtered to **short responses (8-80 words)**, with the tiny
teacher repeated **20×**. A short continued-pretraining run on it should:

- make `<END>` a frequent, expected token after a short answer (fixes symptom 3),
- bias answers toward short and direct (fixes symptom 1),
- amplify the teacher persona so `Who are you?` has an answer.

## 5. Honest caveats about the current 18H corpus (read before training)

The corpus as built is faithful to the spec but has two weaknesses that may cap
the gains. These are **observations, not code changes** — decide if you want to
rebuild with different flags first.

1. **Giant USER instructions survive the filter.** The 8-80 word filter checks
   the **response only**. Measured on `corpus_chat_only.txt`:
   - 38,910 blocks still contain `Input:`,
   - 6.7% of blocks have a USER instruction longer than 80 words,
   - longest instruction = **3,804 words**.
   This keeps re-teaching "USER turns contain big raw text," partly working
   against the goal. A response-only filter cannot catch this.

2. **The teacher is still a minority.** Counts: 5,000 teacher examples
   (250 × 20) vs **113,286** instruction examples (37,762 × 3). Even with the
   20× boost, the crisp-teacher style is ~4% of examples, so the broad Alpaca
   style still dominates.

If 18H underperforms after training, the most likely levers (via existing flags,
no code change) are: cap the instruction set
(`--max-instruction-examples`), lower `--instruction-repeat`, and/or raise
`--teacher-repeat`, so the short, clean teacher style dominates and the long
`Input:`-style turns are diluted.

## 6. How this was verified

- `python 18H_chat_only_pretraining/eval_chat_only.py --cpu --no-strict`
  (current base output in §1).
- Tokenizer single-token check via `tokenizer/bpe_io.load_tokenizer` (§2).
- `corpus_chat_mixed_expanded.txt` inspection + `expanded_corpus_report.json`
  for the 20% chat ratio (§3).
- Direct measurement of instruction lengths in `corpus_chat_only.txt` (§5).
