# Marshmello — Broad SFT Experiment (500 steps)

- Run: `run_broad_sft_500_20260621T102749Z`
- Base: `best_18j_routing.pt`
- Data: `18K_general_benchmark/data/general_train.jsonl` (9304 rows)
- Train: 500 steps, lr 5e-7 → `18B_marshmello_instruct/checkpoints/latest.pt`
- **Deploy: NO** — keep `best_18j_routing.pt`

## Dual benchmark — before vs after

| Metric | Before (best_18j) | After broad SFT (latest) | Δ |
|---|---:|---:|---:|
| **18J routing** | **18.0%** | 17.0% | −1.0pt |
| 18J concept | 33.0% | 25.0% | −8.0pt |
| **18K domain score** | **21.8%** | **12.0%** | **−9.8pt** |
| 18K hallucination | 68.2% | 82.2% | +14.0pt |
| 18K token overlap | 10.5% | 9.6% | −0.9pt |

## 18J by domain (routing)

| Domain | Before | After |
|---|---:|---:|
| ai_basics | 32.5% | 30.0% |
| databases | 10.0% | 13.3% |
| transformers_llms | 6.7% | 3.3% |

## 18K by bucket (domain score)

| Bucket | Before | After |
|---|---:|---:|
| ai | 23.7% | 13.9% |
| databases | 24.6% | 13.4% |
| programming | 19.8% | 10.8% |
| system_design | 20.9% | 9.5% |
| general_knowledge | 20.1% | 12.4% |

## Verdict

**Failed gates:** 18K **collapsed** (not improved); 18J slightly down.

Broad SFT at 500 steps on 9304 general rows **regressed** both benchmarks vs `best_18j_routing`. Same pattern as core SFT and micro-patch: 45M cannot absorb large heterogeneous SFT without hurting what teacher learned.

## Next options

1. **Do not use** `latest.pt` — deploy stays `best_18j_routing.pt`
2. If retry broad SFT: much lighter (≤100–150 steps, lr 2e-7) with strict 18J+18K gates
3. **300M** becomes more justified: general + routing both need capacity; 45M plateaus/regresses on broad data

Archive: `reports/archive/run_broad_sft_500_20260621T102749Z/`
