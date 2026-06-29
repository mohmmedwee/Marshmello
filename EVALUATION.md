# Marshmello Evaluation

Marshmello uses internal educational benchmarks to compare checkpoints and training recipes. These benchmarks are designed for this project; they are not commercial LLM benchmarks and should not be compared to public leaderboard scores.

## Current 55M Results

| Benchmark | Marshmello-55M |
| --------- | -------------: |
| 18J Core Routing | 18% |
| 18K General Domain Score | 22.5% |
| 18K Hallucination | 64.2% |

The 18J result comes from the best core routing checkpoint family. The 18K score and hallucination rate come from the best teacher-family result in the current 18K comparison.

## Phase 18J: Core Concept Routing

18J is a core concept routing benchmark. It asks whether a checkpoint routes a prompt to the right small set of project concepts, especially around:

- AI and machine learning basics
- Transformers and language models
- Databases and SQL

The benchmark is intentionally narrow. It is useful for checking whether a model learned the project’s core educational concepts and whether later tuning damaged those concepts.

What 18J measures:

- Concept routing accuracy
- Whether short answers land in the right domain
- Regression after teacher SFT or routing-specific fine-tuning

What 18J does not measure:

- Broad assistant usefulness
- General factual coverage
- Long-form reasoning
- Production safety

## Phase 18K: General Assistant Benchmark

18K is a general assistant benchmark with 500 held-out examples. It covers five broad buckets:

- AI / ML
- Databases / SQL
- Programming
- System design / DevOps
- General knowledge / writing / daily life

18K is designed to answer a different question from 18J: does the checkpoint behave more like a general educational assistant across broader topics?

Metrics include:

- Domain score
- Keyword recall
- Token overlap
- Hallucination rate
- Empty response rate
- Repetition rate

## Why 18J and 18K Measure Different Things

18J is narrow and routing-focused. A checkpoint can do better on 18J by producing short, recognizable answers for a small set of core concepts.

18K is broader and assistant-focused. A checkpoint must answer across more domains and avoid confident wrong answers, repetition, and empty output.

Because of that, improvements can diverge:

- A routing-tuned checkpoint can preserve 18J while still failing broad assistant questions.
- A broad SFT run can look more general in training but regress on both 18J and 18K.
- A checkpoint should not be promoted unless the target benchmark improves without unacceptable regression elsewhere.

## Why Broad SFT Is Judged by 18K, Not 18J

Broad SFT is meant to improve general assistant behavior. Its primary benchmark is therefore 18K.

18J remains important as a regression check. If broad SFT improves 18K but destroys 18J, it may still be a poor educational Marshmello checkpoint. But if broad SFT does not improve 18K, then it has failed its main purpose even if 18J stays flat.

The current broad SFT experiment regressed the 55M line:

| Metric | Before Broad SFT | After Broad SFT |
| ------ | ---------------: | --------------: |
| 18J routing | 18.0% | 17.0% |
| 18K domain score | 21.8% | 12.0% |
| 18K hallucination | 68.2% | 82.2% |

## Current 55M Plateau

The 55M-class model is useful and stable as an educational baseline, but current experiments suggest it is capacity-limited:

- Best 18J routing is 18%.
- Best 18K domain score is 22.5%.
- Broad SFT at 500 steps regressed both general score and hallucination.
- Additional narrow patches did not create a durable improvement.

The practical conclusion is to stop pushing more heterogeneous SFT into the same small model unless it passes both 18J and 18K gates.

## Why Phase 19A Exists

Phase 19A tests whether more parameters are the right next lever.

The `large_300m` config increases capacity to 268,834,816 parameters while keeping the project architecture and training code familiar. It is a smoke/benchmark phase first: prove the larger model can instantiate, run forward/training/generation passes, fit in memory, and save checkpoints before spending time on long training.

Success for Phase 19A is not just training speed. The real gate is whether a properly trained 300M checkpoint can beat the 55M plateau on 18J and 18K without hiding regressions.
