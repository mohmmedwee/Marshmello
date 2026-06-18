#!/usr/bin/env python3
"""Build a larger clean raw+chat corpus for continued Marshmello pretraining."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_ROOT.parent
PHASE13_ROOT = PROJECT_ROOT / "13_gpt_pretraining"
sys.path.insert(0, str(PHASE13_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "09_bpe_tokenizer_demo"))

from tokenizer.bpe_io import load_tokenizer  # noqa: E402

DEFAULT_BASE_CORPUS = PHASE13_ROOT / "data" / "corpus.txt"
DEFAULT_CHAT_DATA = PROJECT_ROOT / "17_instruction_dataset" / "processed" / "chat.jsonl"
DEFAULT_TOKENIZER = PHASE13_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_OUTPUT = PHASE13_ROOT / "data" / "corpus_chat_mixed_expanded.txt"
DEFAULT_REPORT = PHASE_ROOT / "reports" / "expanded_corpus_report.json"

DOMAIN_SPECS: dict[str, list[tuple[str, str, str, str]]] = {
    "ai_basics": [
        ("artificial intelligence", "software that performs tasks requiring useful reasoning", "classify support requests", "check outputs against clear examples"),
        ("planning", "choosing a sequence of actions before execution", "organize a multi-step workflow", "keep goals and constraints explicit"),
        ("reasoning", "combining facts to reach a supported conclusion", "compare several technical options", "separate evidence from assumptions"),
        ("knowledge representation", "storing facts in a form software can use", "connect entities and relationships", "keep names and meanings consistent"),
        ("search", "exploring possible states until a useful answer is found", "find a route through many choices", "limit wasted work with good heuristics"),
        ("computer vision", "extracting useful information from images", "detect objects in a photograph", "test lighting and camera changes"),
        ("language processing", "reading or producing human language", "summarize a short document", "preserve the original meaning"),
        ("recommendation", "ranking items for a user or situation", "suggest relevant technical articles", "measure usefulness instead of clicks alone"),
        ("AI safety", "reducing harmful or unreliable system behavior", "block unsafe actions", "combine automated checks with human review"),
        ("evaluation", "measuring whether an AI system solves its intended task", "compare answers on held-out prompts", "track quality and failure cases"),
    ],
    "machine_learning": [
        ("supervised learning", "learning from inputs paired with target answers", "predict a category from labeled examples", "keep validation data separate"),
        ("unsupervised learning", "finding structure without target labels", "group similar customer records", "inspect whether groups are meaningful"),
        ("training data", "examples used to adjust model parameters", "teach a classifier from past cases", "remove duplicates and leakage"),
        ("validation", "checking model quality during development", "choose a learning rate", "avoid tuning on the final test set"),
        ("loss functions", "numeric measures of prediction error", "guide gradient updates", "match the loss to the real task"),
        ("gradient descent", "updating weights to reduce loss", "train a neural network", "use stable step sizes"),
        ("overfitting", "memorizing training details instead of general patterns", "recognize falling train loss with weak validation", "use simpler models or better data"),
        ("embeddings", "vectors that represent related items near each other", "compare words or products", "evaluate meaning in the target domain"),
        ("classification", "assigning an input to a known class", "detect spam messages", "measure precision and recall"),
        ("regression", "predicting a numeric value", "estimate delivery time", "check errors across different ranges"),
    ],
    "databases": [
        ("relational tables", "structured rows connected by keys", "store customers and orders", "enforce valid relationships"),
        ("database indexes", "lookup structures that speed row access", "find an account by identifier", "balance faster reads with write cost"),
        ("SQL queries", "declarative requests for relational data", "filter recent transactions", "select only needed columns"),
        ("transactions", "groups of changes committed as one unit", "move value between accounts", "rollback all changes on failure"),
        ("normalization", "organizing fields to reduce duplication", "separate orders from customer details", "avoid unnecessary joins in hot paths"),
        ("query plans", "execution steps selected by the database", "inspect a slow report", "measure plans with realistic data"),
        ("replication", "copying data to additional servers", "serve reads from replicas", "monitor lag and failover"),
        ("backups", "recoverable copies of stored data", "restore after accidental deletion", "test recovery instead of assuming it works"),
        ("constraints", "rules that reject invalid records", "prevent duplicate email addresses", "keep rules close to the data"),
        ("caching", "storing frequent results in faster memory", "reduce repeated database reads", "expire stale values safely"),
    ],
    "software_engineering": [
        ("modular design", "dividing software into focused components", "separate billing from notifications", "keep interfaces small"),
        ("testing", "checking behavior with repeatable examples", "verify an API response", "cover failures as well as success"),
        ("code review", "examining changes before integration", "catch a risky migration", "focus on correctness and maintainability"),
        ("version control", "tracking source changes over time", "develop a feature on a branch", "write clear commits"),
        ("continuous integration", "running automated checks for each change", "test a pull request", "keep feedback fast"),
        ("deployment", "releasing software to an environment", "ship a service update", "support rollback and monitoring"),
        ("observability", "using logs metrics and traces to understand systems", "diagnose a slow request", "record useful context without secrets"),
        ("refactoring", "improving internal structure without changing behavior", "simplify duplicated logic", "protect behavior with tests"),
        ("reliability", "keeping software correct and available", "handle a dependency timeout", "design explicit fallbacks"),
        ("documentation", "explaining behavior and decisions for users and maintainers", "describe an API contract", "update docs with code"),
    ],
    "python": [
        ("functions", "reusable blocks that accept inputs and return results", "normalize a string", "keep each function focused"),
        ("lists", "ordered mutable collections", "store a batch of records", "avoid changing a list while iterating"),
        ("dictionaries", "key-value mappings with fast lookup", "count words in text", "choose stable keys"),
        ("classes", "definitions for objects with state and behavior", "represent a training configuration", "prefer simple data structures when enough"),
        ("exceptions", "signals for errors that callers can handle", "report invalid input", "catch only errors you understand"),
        ("iterators", "objects that produce values one at a time", "stream a large file", "avoid loading unnecessary data"),
        ("context managers", "structured setup and cleanup around resources", "open a file safely", "release resources on failure"),
        ("type hints", "annotations describing expected values", "document a public function", "combine hints with runtime tests"),
        ("unit tests", "small checks for isolated behavior", "test a parser edge case", "use descriptive assertions"),
        ("pathlib", "object-oriented tools for filesystem paths", "build a project-relative file path", "avoid platform-specific separators"),
    ],
    "apis": [
        ("HTTP requests", "messages sent between network clients and servers", "fetch a user record", "set timeouts and handle failures"),
        ("REST resources", "domain objects addressed through URLs", "manage project records", "use consistent nouns and status codes"),
        ("JSON payloads", "structured text used to exchange data", "send configuration values", "validate required fields"),
        ("authentication", "proving the identity of a caller", "accept a signed token", "protect credentials"),
        ("authorization", "deciding what an authenticated caller may do", "limit an editor to one project", "deny access by default"),
        ("pagination", "returning large result sets in smaller pages", "list audit events", "use stable ordering"),
        ("rate limits", "controlling request volume", "protect a public endpoint", "return clear retry guidance"),
        ("idempotency", "making repeated requests produce one final effect", "retry a payment request", "store idempotency keys"),
        ("API versioning", "evolving contracts without breaking clients", "add a new response format", "document migration paths"),
        ("webhooks", "server callbacks for asynchronous events", "notify a client after completion", "verify signatures and retries"),
    ],
    "docker": [
        ("container images", "read-only packages containing an application and dependencies", "ship a Python service", "keep images small and reproducible"),
        ("containers", "isolated processes started from images", "run the same service in testing and production", "store persistent data outside the container"),
        ("Dockerfiles", "instructions for building container images", "install dependencies in layers", "pin important versions"),
        ("image layers", "cached filesystem changes in an image", "reuse dependency installation", "order steps for effective caching"),
        ("volumes", "persistent storage mounted into containers", "keep database files", "manage permissions and backups"),
        ("networks", "virtual connections between containers", "connect an API to a database", "expose only required ports"),
        ("environment variables", "runtime configuration passed to a container", "select a service endpoint", "keep secrets out of images"),
        ("health checks", "commands that report container readiness", "detect a stuck web service", "check real dependencies carefully"),
        ("Docker Compose", "a file describing several local services", "start an API database and cache", "keep development settings explicit"),
        ("container security", "reducing privileges and vulnerable packages", "run as a non-root user", "scan and update base images"),
    ],
    "cybersecurity": [
        ("encryption", "transforming data so unauthorized readers cannot use it", "protect stored credentials", "manage keys separately"),
        ("hashing", "creating a fixed fingerprint from data", "verify file integrity", "use password-specific algorithms for passwords"),
        ("authentication", "confirming a user or service identity", "verify a login", "support strong second factors"),
        ("authorization", "limiting actions after identity is known", "restrict administrative settings", "apply least privilege"),
        ("input validation", "checking external data before use", "reject malformed request fields", "validate type length and format"),
        ("secure coding", "building software that resists common attacks", "parameterize a database query", "review trust boundaries"),
        ("network segmentation", "separating systems into controlled zones", "isolate production databases", "allow only required traffic"),
        ("security logging", "recording events useful for detection and investigation", "track failed logins", "avoid logging secrets"),
        ("patch management", "updating vulnerable software in a controlled way", "replace an unsafe dependency", "test and deploy fixes promptly"),
        ("incident response", "containing and learning from security events", "handle a leaked credential", "preserve evidence and communicate clearly"),
    ],
    "transformers_llms": [
        ("tokens", "text pieces represented by numeric identifiers", "encode a user prompt", "keep training and inference tokenizers identical"),
        ("BPE tokenization", "learning common subword pieces from text", "represent rare technical words", "train it on the intended data distribution"),
        ("embeddings", "learned vectors for token meaning and position", "represent an input sequence", "match embedding rows to vocabulary size"),
        ("attention", "weighting earlier tokens by relevance", "connect a question to useful context", "apply the causal mask during generation"),
        ("transformer blocks", "attention and feed-forward layers joined by residual paths", "build a decoder-only model", "use normalization for stable training"),
        ("causal language modeling", "predicting each next token from previous tokens", "pretrain on raw text", "prevent future-token leakage"),
        ("context windows", "limited token sequences visible to the model", "answer from a prompt and recent history", "truncate without losing key instructions"),
        ("pretraining", "learning broad language patterns before task tuning", "train on technical paragraphs", "use enough diverse clean text"),
        ("instruction tuning", "training on user requests and assistant answers", "teach direct response behavior", "start from a capable base model"),
        ("generation controls", "settings that shape token sampling", "reduce repetitive output", "evaluate with fixed prompts and seeds"),
    ],
}

CONTEXTS = (
    "a small internal service",
    "a production application",
    "a learning project",
    "an analytics workflow",
    "a customer-facing system",
    "a batch processing job",
    "a security-sensitive environment",
    "a team codebase",
    "a local development setup",
    "a high-traffic platform",
    "a data quality review",
    "a model evaluation pipeline",
)

GOALS = (
    "make behavior easier to understand",
    "reduce avoidable errors",
    "improve response quality",
    "keep operations predictable",
    "support safe incremental changes",
    "measure results with clear evidence",
    "serve users with lower latency",
    "protect important data",
    "simplify maintenance",
    "scale without hiding failures",
)

Pair = tuple[str, str]


def normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def load_base_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = [normalize_block(part) for part in re.split(r"\n\s*\n", text)]
    return [block for block in blocks if block]


def load_chat_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            text = normalize_block(str(record.get("text", "")))
            domain = normalize_block(str(record.get("domain", "general"))) or "general"
            if text:
                records.append((text, domain))
    if not records:
        raise ValueError(f"No chat examples found in {path}")
    return records


def synthetic_paragraph(
    domain: str,
    primary: tuple[str, str, str, str],
    secondary: tuple[str, str, str, str],
    context: str,
    goal: str,
    variant: int,
) -> str:
    name, definition, example, caution = primary
    second_name, second_definition, second_example, second_caution = secondary
    templates = (
        (
            f"{name.capitalize()} means {definition}. In {context}, teams can use it to {example} "
            f"and {goal}. It often works beside {second_name}, which means {second_definition}. "
            f"A practical workflow starts with a small example, measures the result, and then expands carefully. "
            f"Engineers should {caution}, while also remembering to {second_caution}."
        ),
        (
            f"A useful lesson in {domain.replace('_', ' ')} is {name}. It is {definition}, and a clear use case "
            f"is to {example} in {context}. The related idea of {second_name} helps because it is {second_definition}. "
            f"Together they can {goal} when the inputs, expected behavior, and checks are written down. "
            f"Good practice is to {caution} and {second_caution}."
        ),
        (
            f"When working in {context}, {name} provides {definition}. One concrete task is to {example}. "
            f"The design can also include {second_name}, described as {second_definition}, to {second_example}. "
            f"This combination should be tested on normal cases and failures so teams can {goal}. "
            f"To keep the result reliable, {caution} and {second_caution}."
        ),
        (
            f"{name.capitalize()} is important for teams that want to {goal}. The concept refers to {definition}. "
            f"For example, a developer may {example} while building {context}. {second_name.capitalize()} adds "
            f"another useful tool because it is {second_definition}. Start with observable requirements, then "
            f"measure the outcome, {caution}, and {second_caution}."
        ),
        (
            f"In {domain.replace('_', ' ')}, a common pattern combines {name} with {second_name}. "
            f"The first is {definition}; the second is {second_definition}. A team might {example} and then "
            f"{second_example} inside {context}. This keeps the work connected to a real goal and can {goal}. "
            f"The implementation remains dependable when engineers {caution} and {second_caution}."
        ),
    )
    return templates[variant % len(templates)]


def generate_synthetic_blocks(target_words: int) -> tuple[list[tuple[str, str]], Counter[str]]:
    blocks: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    domains = list(DOMAIN_SPECS)
    index = 0

    while sum(counts.values()) < target_words:
        domain = domains[index % len(domains)]
        topics = DOMAIN_SPECS[domain]
        primary_idx = (index // len(domains)) % len(topics)
        secondary_idx = (
            primary_idx
            + 1
            + (index // (len(domains) * len(topics))) % (len(topics) - 1)
        ) % len(topics)
        context_idx = (index // 7 + primary_idx * 3) % len(CONTEXTS)
        goal_idx = (index // 11 + secondary_idx * 5) % len(GOALS)
        variant = (index // 13 + primary_idx + secondary_idx) % 5
        block = synthetic_paragraph(
            domain,
            topics[primary_idx],
            topics[secondary_idx],
            CONTEXTS[context_idx],
            GOALS[goal_idx],
            variant,
        )
        normalized = normalize_block(block)
        if normalized not in seen:
            seen.add(normalized)
            words = word_count(normalized)
            blocks.append((domain, normalized))
            counts[domain] += words
        index += 1
        if index > 2_000_000:
            raise RuntimeError("Could not generate enough unique synthetic paragraphs")
    return blocks, counts


def select_chat_blocks(
    records: list[tuple[str, str]],
    *,
    target_words: int,
    seed: int,
) -> tuple[list[tuple[str, str]], Counter[str]]:
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    selected: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    total = 0
    for text, domain in shuffled:
        selected.append((f"chat_{domain}", text))
        words = word_count(text)
        counts[f"chat_{domain}"] += words
        total += words
        if total >= target_words:
            break
    if total < target_words:
        raise ValueError(
            f"Chat data has only {total:,} words, below requested {target_words:,}"
        )
    return selected, counts


def merge_pair(symbols: list[str], pair: Pair, replacement: str) -> list[str]:
    first, second = pair
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if (
            index < len(symbols) - 1
            and symbols[index] == first
            and symbols[index + 1] == second
        ):
            merged.append(replacement)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


def merge_rank_map(bpe) -> dict[Pair, int]:
    ranks: dict[Pair, int] = {}
    for rank, pair in enumerate(bpe.merges):
        ranks.setdefault(pair, rank)
    return ranks


def encode_word_count(word: str, bpe, ranks: dict[Pair, int]) -> int:
    symbols = [*word, bpe.END]
    while len(symbols) > 1:
        best_pair: Pair | None = None
        best_rank: int | None = None
        for pair in zip(symbols, symbols[1:]):
            rank = ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_pair = pair
                best_rank = rank
        if best_pair is None:
            break
        symbols = merge_pair(symbols, best_pair, best_pair[0] + best_pair[1])
    return sum(token in bpe.stoi for token in symbols)


def estimate_bpe_tokens(
    blocks: list[tuple[str, str]],
    *,
    tokenizer_path: Path,
    total_words: int,
    sample_words: int = 50_000,
) -> dict[str, object]:
    bpe = load_tokenizer(tokenizer_path)
    ranks = merge_rank_map(bpe)
    known = {token for token in bpe.vocab if len(token) == 1}
    cache: dict[str, int] = {}
    sampled_words = 0
    sampled_tokens = 0

    for _, block in blocks:
        for word in block.split():
            safe_word = "".join(char for char in word if char in known)
            if safe_word:
                token_count = cache.get(safe_word)
                if token_count is None:
                    token_count = encode_word_count(safe_word, bpe, ranks)
                    cache[safe_word] = token_count
                sampled_tokens += token_count
            sampled_words += 1
            if sampled_words >= sample_words:
                ratio = sampled_tokens / sampled_words
                return {
                    "estimated_tokens": round(total_words * ratio),
                    "sample_words": sampled_words,
                    "sample_tokens": sampled_tokens,
                    "tokens_per_word": round(ratio, 4),
                    "tokenizer": str(tokenizer_path),
                }
    ratio = sampled_tokens / max(sampled_words, 1)
    return {
        "estimated_tokens": round(total_words * ratio),
        "sample_words": sampled_words,
        "sample_tokens": sampled_tokens,
        "tokens_per_word": round(ratio, 4),
        "tokenizer": str(tokenizer_path),
    }


def build_report(
    *,
    output_path: Path,
    target_words: int,
    requested_chat_ratio: float,
    blocks: list[tuple[str, str]],
    base_word_count: int,
    synthetic_counts: Counter[str],
    chat_counts: Counter[str],
    chat_examples_count: int,
    tokenizer_path: Path,
) -> dict[str, object]:
    distribution = Counter({"base_corpus": base_word_count})
    distribution.update(synthetic_counts)
    distribution.update(chat_counts)
    total_words = sum(distribution.values())
    chat_words = sum(chat_counts.values())
    normalized_blocks = [normalize_block(text).casefold() for _, text in blocks]
    duplicate_blocks = len(normalized_blocks) - len(set(normalized_blocks))
    duplicate_ratio = duplicate_blocks / len(normalized_blocks) if normalized_blocks else 0.0
    estimated = estimate_bpe_tokens(
        blocks,
        tokenizer_path=tokenizer_path,
        total_words=total_words,
    )
    return {
        "output": str(output_path),
        "target_words": target_words,
        "total_words": total_words,
        "requested_chat_ratio": requested_chat_ratio,
        "actual_chat_ratio": round(chat_words / total_words, 6),
        "domain_distribution_words": dict(sorted(distribution.items())),
        "domain_distribution_percent": {
            domain: round(words / total_words, 6)
            for domain, words in sorted(distribution.items())
        },
        "chat_examples_count": chat_examples_count,
        "total_blocks": len(blocks),
        "duplicate_blocks": duplicate_blocks,
        "duplicate_ratio": round(duplicate_ratio, 6),
        "estimated_bpe_tokens": estimated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 18G expanded corpus.")
    parser.add_argument("--base-corpus", type=Path, default=DEFAULT_BASE_CORPUS)
    parser.add_argument("--chat-data", type=Path, default=DEFAULT_CHAT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-words", type=int, default=5_000_000)
    parser.add_argument("--chat-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.target_words < 100_000:
        raise ValueError("--target-words must be at least 100000")
    if not 0.0 < args.chat_ratio < 1.0:
        raise ValueError("--chat-ratio must be between 0 and 1")
    for label, path in (
        ("base corpus", args.base_corpus),
        ("chat data", args.chat_data),
        ("tokenizer", args.tokenizer),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    base_blocks = load_base_blocks(args.base_corpus)
    base_words = sum(word_count(block) for block in base_blocks)
    raw_target = round(args.target_words * (1.0 - args.chat_ratio))
    synthetic_target = max(0, raw_target - base_words)

    print("Phase 18G: expanded corpus builder")
    print("=" * 64)
    print(f"Target words:          {args.target_words:,}")
    print(f"Requested chat ratio:  {args.chat_ratio:.1%}")
    print(f"Base corpus words:     {base_words:,}")
    print(f"Synthetic words needed:{synthetic_target:>12,}")

    synthetic_blocks, synthetic_counts = generate_synthetic_blocks(synthetic_target)
    raw_words = base_words + sum(synthetic_counts.values())
    chat_target = round(raw_words * args.chat_ratio / (1.0 - args.chat_ratio))
    chat_records = load_chat_records(args.chat_data)
    chat_blocks, chat_counts = select_chat_blocks(
        chat_records,
        target_words=chat_target,
        seed=args.seed,
    )

    tagged_blocks: list[tuple[str, str]] = [
        ("base_corpus", block) for block in base_blocks
    ]
    tagged_blocks.extend(synthetic_blocks)
    tagged_blocks.extend(chat_blocks)
    random.Random(args.seed).shuffle(tagged_blocks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n\n".join(text for _, text in tagged_blocks).strip() + "\n",
        encoding="utf-8",
    )

    report = build_report(
        output_path=args.output,
        target_words=args.target_words,
        requested_chat_ratio=args.chat_ratio,
        blocks=tagged_blocks,
        base_word_count=base_words,
        synthetic_counts=synthetic_counts,
        chat_counts=chat_counts,
        chat_examples_count=len(chat_blocks),
        tokenizer_path=args.tokenizer,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Synthetic blocks:      {len(synthetic_blocks):,}")
    print(f"Chat examples:         {len(chat_blocks):,}")
    print(f"Total blocks:          {report['total_blocks']:,}")
    print(f"Total words:           {report['total_words']:,}")
    print(f"Actual chat ratio:     {report['actual_chat_ratio']:.2%}")
    print(f"Duplicate ratio:       {report['duplicate_ratio']:.2%}")
    estimate = dict(report["estimated_bpe_tokens"])
    print(f"Estimated BPE tokens:  {estimate['estimated_tokens']:,}")
    print(f"Output:                {args.output}")
    print(f"Report:                {args.report_output}")


if __name__ == "__main__":
    main()
