"""Phase 16 evaluation prompt suite."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalPrompt:
    prompt: str
    domain: str
    label: str


PROMPT_SUITE: tuple[EvalPrompt, ...] = (
    EvalPrompt("Database systems", "databases", "Database systems"),
    EvalPrompt("Artificial intelligence", "artificial_intelligence", "Artificial intelligence"),
    EvalPrompt("Machine learning", "machine_learning", "Machine learning"),
    EvalPrompt("Software engineering", "software_engineering", "Software engineering"),
    EvalPrompt("Python APIs", "software_engineering", "Python APIs"),
    EvalPrompt("Distributed systems", "distributed_systems", "Distributed systems"),
)

MODEL_ALIASES: dict[str, str] = {
    "default": "Marshmello-8M",
    "large_50m": "Marshmello-45M",
}

EVAL_CONFIGS: tuple[str, ...] = ("default", "large_50m")

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "databases": (
        "database",
        "sql",
        "query",
        "table",
        "index",
        "transaction",
        "schema",
        "row",
        "column",
        "postgres",
        "mysql",
    ),
    "artificial_intelligence": (
        "artificial intelligence",
        "ai",
        "reasoning",
        "agent",
        "knowledge",
        "robotics",
        "planning",
        "search",
        "intelligence",
    ),
    "machine_learning": (
        "machine learning",
        "model",
        "training",
        "dataset",
        "loss",
        "gradient",
        "neural",
        "supervised",
        "unsupervised",
        "validation",
    ),
    "software_engineering": (
        "software",
        "engineering",
        "api",
        "code",
        "test",
        "deploy",
        "repository",
        "python",
        "function",
        "module",
        "review",
    ),
    "distributed_systems": (
        "distributed",
        "cluster",
        "replication",
        "shard",
        "consensus",
        "latency",
        "partition",
        "scale",
        "node",
        "fault",
    ),
}
