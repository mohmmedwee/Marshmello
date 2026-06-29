---
license: apache-2.0
language:
- en
task_categories:
- text-generation
- question-answering
tags:
- instruction-tuning
- educational
- marshmello
size_categories:
- 1K<n<10K
---

# Marshmello-SFT

Marshmello-SFT is an educational instruction-tuning dataset used by the Marshmello model family. It is designed to support small-model SFT experiments and internal evaluation, not to serve as a broad production assistant dataset.

This dataset card is a draft for Hugging Face publication. It should be updated with the exact uploaded file list and dataset commit before publishing.

## Dataset Description

| Field | Value |
| ----- | ----- |
| Dataset name | Marshmello-SFT |
| Examples | 9,804 |
| Format | JSONL instruction/response examples |
| Primary language | English |
| Intended model family | Marshmello |
| Intended use | Educational SFT and benchmark construction |

The current split used by Phase 18K is:

| Split | Examples |
| ----- | -------: |
| Train pool | 9,304 |
| Held-out eval | 500 |
| Total | 9,804 |

## Domains

Marshmello-SFT covers educational examples across:

- AI and machine learning
- Deep learning
- Transformers and language models
- Math basics and statistics
- Databases, data structures, and SQL
- Python and programming
- Algorithms
- Software engineering and web development
- System design, Linux, networking, DevOps, and cybersecurity
- General science, history, geography, study skills, writing, and daily life

## Generation Methodology

The dataset is assembled through project-local builders and curation scripts. It combines short educational instruction/response examples, teacher SFT data, routing-oriented examples, and broader assistant-style examples used for Phase 18K.

The dataset is meant to test how small Marshmello checkpoints respond to incremental SFT. It is intentionally modest in size so the full pipeline remains understandable and runnable in a learning environment.

## Validation

Validation currently focuses on project-level checks:

- JSONL row counts.
- Non-overlapping Phase 18K train/eval split.
- Bucketed held-out evaluation set with 500 examples.
- Internal benchmark reports for domain score, hallucination, repetition, and empty outputs.

The dataset has not been exhaustively audited for every factual claim, bias, or unsafe output pattern.

## Intended Educational Use

Use this dataset to:

- Teach instruction tuning mechanics.
- Compare Marshmello checkpoints.
- Study overfitting and benchmark regression in small models.
- Build transparent model and dataset documentation.

Do not use this dataset as the sole data source for a production assistant.

## Limitations

- English-focused.
- Small and educational, not web-scale.
- Some examples are generated or template-assisted.
- Coverage is uneven across domains.
- May contain factual mistakes or overly simple answers.
- Does not provide safety alignment sufficient for public production deployment.

## Ethical Notes

Models trained on this dataset can still hallucinate, repeat text, or answer incorrectly. Any public use should add separate safety review, filtering, monitoring, and domain-specific evaluation.
