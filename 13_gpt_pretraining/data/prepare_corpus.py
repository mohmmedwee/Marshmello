"""
Write data/corpus.txt with selectable domain mix.

Mix options (--mix):
  classic — phase 08 Shakespeare-adjacent prose only
  tech    — AI / ML / databases / software engineering only (no classic text)
  mixed   — both domains shuffled together

Corpus assembly:
  - each unique paragraph repeated --repeat-per-domain times (default 20)
  - paragraphs shuffled before write
  - optional --dedupe-lines removes duplicate paragraphs
  - section markers stripped; whitespace normalized
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_ROOT.parent
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "08_word_level_language_model"))

from corpus import PASSAGES  # noqa: E402

from config import CORPUS_PATH, DEFAULT_CONFIG  # noqa: E402

DATA_MIX_OPTIONS = ("classic", "tech", "mixed")

MIX_ALIASES = {
    "classic_literature": "classic",
    "technical_ai": "tech",
}

SECTION_MARKER_PATTERNS: tuple[str, ...] = (
    r"^=== DOMAIN: .+ ===\s*$",
    r"^=== Topic: .+ ===\s*$",
    r"^=== Book \d+ ===\s*$",
    r"^Scene \d+\s*$",
    r"^=== Supplement \d+ ===\s*$",
    r"^=== Technical \d+ ===\s*$",
)

TECHNICAL_PASSAGES: list[tuple[str, str]] = [
    (
        "artificial_intelligence",
        """
        Artificial intelligence builds systems that perceive, reason, and act.
        Search algorithms explore state spaces. Knowledge graphs store entities and
        relations. Planning selects actions toward goals. Robotics combines perception,
        control, and learning. AI safety studies alignment, robustness, and oversight.
        Evaluation benchmarks measure reasoning, coding, and multimodal understanding.
        """,
    ),
    (
        "machine_learning",
        """
        Machine learning trains models on data instead of hand-written rules.
        Supervised learning maps inputs to labels with cross-entropy or regression loss.
        Unsupervised learning discovers clusters, factors, or latent structure.
        Reinforcement learning optimizes policies from reward signals over time.
        Feature engineering, regularization, and validation splits reduce overfitting.
        Neural networks use layers, activations, and optimizers such as AdamW.
        """,
    ),
    (
        "deep_learning",
        """
        Deep learning stacks nonlinear layers to learn hierarchical representations.
        Convolutional networks excel at images. Recurrent networks model sequences but
        transformers dominate language modeling today. Attention computes weighted sums
        over context tokens. Pretraining predicts the next token on large corpora.
        Fine-tuning adapts a pretrained model to classification or generation tasks.
        """,
    ),
    (
        "databases",
        """
        Database systems store durable structured data for applications and analytics.
        Relational databases use tables, rows, columns, and SQL for queries.
        Indexes speed lookups; B-tree and hash indexes serve different access patterns.
        Transactions provide ACID guarantees with locking or multi-version concurrency.
        Normalization reduces redundancy; denormalization can improve read performance.
        Replication and sharding scale reads and writes across machines.
        """,
    ),
    (
        "sql_and_queries",
        """
        SQL selects, filters, joins, groups, and aggregates records.
        Primary keys uniquely identify rows; foreign keys link tables.
        Query planners choose index scans versus full table scans.
        Migrations version schema changes in production safely.
        ORMs map objects to tables but complex reports still need raw SQL.
        """,
    ),
    (
        "software_engineering",
        """
        Software engineering applies structured processes to build reliable systems.
        Requirements define user needs; design documents capture architecture decisions.
        Version control tracks changes; code review improves quality and knowledge sharing.
        Continuous integration runs tests on every commit before deployment.
        Observability combines logs, metrics, and traces to debug production issues.
        Refactoring improves structure without changing external behavior.
        """,
    ),
    (
        "python_and_apis",
        """
        Python packages organize reusable modules with clear public APIs.
        Virtual environments isolate dependencies per project.
        REST APIs expose resources over HTTP with JSON payloads.
        Authentication uses tokens, sessions, or OAuth depending on threat models.
        Documentation explains endpoints, error codes, and example requests.
        """,
    ),
    (
        "systems_and_devops",
        """
        Operating systems schedule processes, manage memory, and expose file systems.
        Containers package applications with dependencies for reproducible deployment.
        Kubernetes orchestrates pods, services, and rolling updates in clusters.
        Infrastructure as code defines servers and networks in versioned templates.
        Load balancers distribute traffic; caches reduce latency for hot data.
        """,
    ),
]


def normalize_mix_name(data_mix: str) -> str:
    data_mix = data_mix.strip().lower()
    data_mix = MIX_ALIASES.get(data_mix, data_mix)
    if data_mix not in DATA_MIX_OPTIONS:
        raise ValueError(f"Unknown mix: {data_mix!r}. Choose from {DATA_MIX_OPTIONS}")
    return data_mix


def normalize_paragraph(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def dedupe_paragraphs(paragraphs: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for paragraph in paragraphs:
        norm = normalize_paragraph(paragraph)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def expand_paragraphs(
    paragraphs: list[str],
    repeat_per_domain: int,
    domain_label: str,
) -> list[tuple[str, str]]:
    """Return (domain, paragraph) pairs with limited per-paragraph repetition."""
    tagged: list[tuple[str, str]] = []
    for paragraph in paragraphs:
        norm = normalize_paragraph(paragraph)
        if not norm:
            continue
        for _ in range(repeat_per_domain):
            tagged.append((domain_label, norm))
    return tagged


def classic_source_paragraphs() -> list[str]:
    return [normalize_paragraph(p) for p in PASSAGES if normalize_paragraph(p)]


def tech_source_paragraphs() -> list[str]:
    return [normalize_paragraph(p) for _, p in TECHNICAL_PASSAGES if normalize_paragraph(p)]


def build_tagged_paragraphs(
    data_mix: str,
    repeat_per_domain: int,
) -> list[tuple[str, str]]:
    data_mix = normalize_mix_name(data_mix)
    tagged: list[tuple[str, str]] = []

    if data_mix in ("classic", "mixed"):
        tagged.extend(
            expand_paragraphs(classic_source_paragraphs(), repeat_per_domain, "classic")
        )
    if data_mix in ("tech", "mixed"):
        for topic, passage in TECHNICAL_PASSAGES:
            norm = normalize_paragraph(passage)
            if not norm:
                continue
            for _ in range(repeat_per_domain):
                tagged.append((f"tech/{topic}", norm))

    return tagged


def clean_corpus(text: str) -> str:
    for pattern in SECTION_MARKER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    text = re.sub(r"^={3,}[^=\n]*={3,}\s*$", "", text, flags=re.MULTILINE)

    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line.strip())
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(" ".join(buffer))

    deduped: list[str] = []
    for paragraph in paragraphs:
        if not deduped or deduped[-1] != paragraph:
            deduped.append(paragraph)
    return "\n\n".join(deduped).strip()


def domain_counts_from_tagged(tagged: list[tuple[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for domain, paragraph in tagged:
        counts[domain] += len(paragraph.split())
    counts["classic"] = counts.get("classic", 0)
    counts["tech"] = sum(v for k, v in counts.items() if k.startswith("tech/"))
    return counts


def print_domain_distribution(counts: Counter[str], file_words: int, data_mix: str) -> None:
    classic = counts.get("classic", 0)
    tech = counts.get("tech", 0)
    total = file_words

    print("Corpus domain distribution:")
    if data_mix in ("classic", "mixed") and classic > 0:
        pct = 100.0 * classic / total if total else 0.0
        print(f"  {'classic':<28} {classic:>8,} words  ({pct:5.1f}%)")
    if data_mix in ("tech", "mixed") and tech > 0:
        pct = 100.0 * tech / total if total else 0.0
        print(f"  {'tech (total)':<28} {tech:>8,} words  ({pct:5.1f}%)")
        for domain, words in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if domain.startswith("tech/") and words > 0:
                sub_pct = 100.0 * words / total if total else 0.0
                label = domain.removeprefix("tech/")
                print(f"    └ {label:<24} {words:>8,} words  ({sub_pct:5.1f}%)")
    print(f"  {'TOTAL (written)':<28} {total:>8,} words")
    print()


def print_data_quality_report(text: str) -> None:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    para_total = len(paragraphs)
    para_unique = len(set(paragraphs))
    line_total = len(lines)
    line_unique = len(set(lines))

    para_dup_ratio = 1.0 - (para_unique / para_total) if para_total else 0.0
    line_dup_ratio = 1.0 - (line_unique / line_total) if line_total else 0.0

    print("Data quality report:")
    print(f"  paragraphs total:   {para_total:,}")
    print(f"  paragraphs unique:  {para_unique:,}")
    print(f"  paragraph dup ratio:{para_dup_ratio:6.1%}")
    print(f"  lines total:        {line_total:,}")
    print(f"  lines unique:       {line_unique:,}")
    print(f"  line dup ratio:     {line_dup_ratio:6.1%}")

    if paragraphs:
        print("  top repeated paragraphs:")
        for snippet, count in Counter(paragraphs).most_common(5):
            if count <= 1:
                break
            preview = snippet[:72] + ("..." if len(snippet) > 72 else "")
            print(f"    ×{count:<3} {preview!r}")
    print()


def build_corpus_for_mix(
    data_mix: str,
    repeat_per_domain: int,
    dedupe_lines: bool,
    seed: int,
) -> tuple[str, Counter[str]]:
    tagged = build_tagged_paragraphs(data_mix, repeat_per_domain)
    domain_counts = domain_counts_from_tagged(tagged)

    paragraphs = [p for _, p in tagged]
    if dedupe_lines:
        paragraphs = dedupe_paragraphs(paragraphs)

    rng = random.Random(seed)
    rng.shuffle(paragraphs)

    raw = "\n\n".join(paragraphs)
    cleaned = clean_corpus(raw)
    return cleaned, domain_counts


def prepare_corpus(
    min_words: int | None = None,
    force: bool = False,
    data_mix: str = "mixed",
    repeat_per_domain: int = 20,
    dedupe_lines: bool = False,
    seed: int | None = None,
) -> Path:
    data_mix = normalize_mix_name(data_mix)
    min_words = min_words or DEFAULT_CONFIG.corpus_min_words
    seed = seed if seed is not None else DEFAULT_CONFIG.seed

    if CORPUS_PATH.exists() and not force:
        return CORPUS_PATH

    cleaned, domain_counts = build_corpus_for_mix(
        data_mix,
        repeat_per_domain=repeat_per_domain,
        dedupe_lines=dedupe_lines,
        seed=seed,
    )

    words = len(cleaned.split())
    raw_domain_total = domain_counts.get("classic", 0) + domain_counts.get("tech", 0)
    if raw_domain_total > 0 and words != raw_domain_total:
        scale = words / raw_domain_total
        for key in list(domain_counts.keys()):
            domain_counts[key] = int(domain_counts[key] * scale)
        domain_counts["classic"] = int(domain_counts.get("classic", 0))
        domain_counts["tech"] = sum(
            v for k, v in domain_counts.items() if k.startswith("tech/")
        )

    if words < min_words:
        print(
            f"Note: corpus has {words:,} words (target min_words={min_words:,}). "
            f"Increase --repeat-per-domain instead of duplicating whole blocks."
        )

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(cleaned, encoding="utf-8")

    print(f"Data mix: {data_mix}")
    print(f"Repeat per domain paragraph: {repeat_per_domain}")
    print(f"Dedupe lines: {dedupe_lines}")
    print(f"Wrote {CORPUS_PATH} ({words:,} words, {len(cleaned):,} chars)")
    print_domain_distribution(domain_counts, words, data_mix)
    print_data_quality_report(cleaned)
    return CORPUS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 13 training corpus")
    parser.add_argument(
        "--mix",
        choices=list(DATA_MIX_OPTIONS),
        default="mixed",
        help="Corpus domain mix: classic | tech | mixed (default: mixed)",
    )
    parser.add_argument(
        "--repeat-per-domain",
        type=int,
        default=20,
        help="Repeat each source paragraph this many times before shuffle (default: 20)",
    )
    parser.add_argument(
        "--dedupe-lines",
        action="store_true",
        help="Remove duplicate paragraphs before shuffle",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild corpus.txt")
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    args = parser.parse_args()

    prepare_corpus(
        min_words=args.min_words,
        force=args.force,
        data_mix=args.mix,
        repeat_per_domain=args.repeat_per_domain,
        dedupe_lines=args.dedupe_lines,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
