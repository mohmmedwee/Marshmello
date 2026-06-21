# Phase 18B — Marshmello-45M-Instruct Fine-tuning

This phase fine-tunes **Marshmello-45M-Base-v2** into
**Marshmello-45M-Instruct** using Phase 17 chat-formatted instruction data.

Do not run this before base pretraining. Base pretraining gives the model
language ability, technical vocabulary, and broad world knowledge. Instruction
tuning teaches prompt-following behavior: how to treat `<USER>` as a request,
how to answer after `<ASSISTANT>`, and how to stop at `<END>`.

SFT without a good base model mostly memorizes answers. It can pass tiny exact
matches while still failing new prompts because the model has not learned enough
language or domain structure underneath.

---

## Inputs

Base checkpoint:

```text
13_gpt_pretraining/checkpoints/large_50m/latest.pt
```

Tokenizer:

```text
13_gpt_pretraining/tokenizer/tokenizer.json
```

Instruction data:

```text
17_instruction_dataset/processed/chat.jsonl
```

---

## SFT Format

```text
<USER> instruction <ASSISTANT> response <END>
```

The training loss:

- ignores user tokens
- trains assistant response tokens
- trains `<END>`
- weights the first assistant answer token higher to improve routing (default 8; use `--first-token-weight 30` for core SFT from a routing checkpoint)

## Deploy checkpoint

Best **18J** core routing on 45M (~18%):

```text
18B_marshmello_instruct/checkpoints/best_18j_routing.pt
```

Do **not** deploy `latest.pt` after failed broad SFT (see `reports/latest_eval_summary.md`).

---

## Train

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --config large_50m \
  --base-checkpoint 13_gpt_pretraining/checkpoints/large_50m/step_001000.pt \
  --steps 500 \
  --lr 1e-5 \
  --freeze-backbone
```

Checkpoint output:

```text
18B_marshmello_instruct/checkpoints/latest.pt
```

The trainer prints train loss, validation loss, tokens/sec, and checkpoint path.
It supports MPS automatically when available. Before training it verifies that
the tokenizer vocab, model vocab, and checkpoint embedding shape match; it also
prints missing/unexpected checkpoint keys and runs an encode/decode sanity check
for `<USER>`, `<ASSISTANT>`, and `<END>`.

Overfit diagnostic:

```bash
python 18B_marshmello_instruct/train_instruct.py \
  --mode overfit \
  --max-examples 20 \
  --steps 300
```

---

## Chat

```bash
python 18B_marshmello_instruct/chat.py \
  --checkpoint 18B_marshmello_instruct/checkpoints/best_18j_routing.pt \
  --prompt "Explain database indexes" \
  --greedy
```

## Dual benchmarks

```bash
python 18J_marshmello_core_sft/evaluate_core_routing.py \
  --checkpoint 18B_marshmello_instruct/checkpoints/best_18j_routing.pt --no-baseline

python 18K_general_benchmark/evaluate_general.py \
  --label best_18j_routing \
  --checkpoint 18B_marshmello_instruct/checkpoints/best_18j_routing.pt
```

---

## Evaluate Base vs Instruct

```bash
python 18B_marshmello_instruct/eval_instruct.py
```

Prompts:

- What is AI?
- Explain database indexes.
- Write a Python function to reverse a string.
- Explain Docker simply.
- What is BPE?

---

## Full Flow

```bash
python 18A_large_pretraining_corpus/build_corpus.py --target-words 1000000
python 13_gpt_pretraining/tokenizer/train_bpe.py
python 13_gpt_pretraining/training/trainer.py --config large_50m --steps 3000

python 18B_marshmello_instruct/train_instruct.py \
  --config large_50m \
  --steps 500 \
  --lr 1e-5 \
  --freeze-backbone
python 18B_marshmello_instruct/chat.py --prompt "Explain database indexes"
```
