#!/usr/bin/env python3
"""
Build the Phase 18I routing-teacher dataset.

After 18H + teacher SFT the model can produce short answers and stop at
``<END>``, but it routes questions to the wrong definition (for example
"What is AI?" -> a tokenizer answer, or "What is a tokenizer?" -> a database
index answer). This dataset fixes routing by giving each key concept many
*paraphrased* questions that all map to a consistent (but not identical) answer,
while keeping the concepts' answer vocabularies disjoint.

That disjointness is the hard-negative signal: AI/identity answers never contain
tokenizer / database words, tokenizer answers never contain database/index
words, and database answers never contain tokenizer words. The build fails if
any answer leaks a competing concept's signature term, so the training data can
never teach the very confusion we are trying to remove.

Answers are written as short, simple subject-verb sentences so the model has
clean, robust targets (the earlier dataset produced on-topic but grammatically
weak output for AI, identity, and attention).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PHASE_ROOT / "data" / "routing_teacher.jsonl"

MIN_RESPONSE_WORDS = 6
MAX_RESPONSE_WORDS = 45
MIN_TOTAL_EXAMPLES = 100
MAX_TOTAL_EXAMPLES = 300

# Generic, contentless suffixes that previous teacher data used. They make every
# concept sound the same and hurt routing, so they are banned outright.
BANNED_PHRASES = (
    "it helps the system answer or act more usefully",
    "it helps the system answer",
    "act more usefully",
)


@dataclass(frozen=True)
class Concept:
    name: str
    domain: str
    expected_count: int
    instructions: tuple[str, ...]
    answers: tuple[str, ...]
    # Signature terms of *competing* concepts that must never appear here.
    forbidden_terms: tuple[str, ...]


AI_INSTRUCTIONS = (
    "What is AI?",
    "Define artificial intelligence.",
    "What does AI mean?",
    "Explain artificial intelligence.",
    "Can you describe AI?",
    "Tell me what AI is.",
    "What is artificial intelligence?",
    "How would you define AI?",
    "Give a short definition of AI.",
    "In simple terms, what is AI?",
    "What do we mean by AI?",
    "Describe what artificial intelligence is.",
    "What is AI in computing?",
    "What is the meaning of AI?",
    "Explain AI in one sentence.",
    "What exactly is artificial intelligence?",
    "Could you explain what AI is?",
    "What is AI, briefly?",
    "Help me understand what AI is.",
    "What is meant by artificial intelligence?",
    "Define AI for me.",
    "What is AI all about?",
    "Summarize what artificial intelligence is.",
    "What is AI technology?",
    "Give me a simple explanation of AI.",
    "What does the term AI refer to?",
    "What is artificial intelligence in plain words?",
    "Explain the concept of AI.",
    "What is AI supposed to be?",
    "Tell me about artificial intelligence.",
    "What is AI, in one line?",
    "How do you explain AI to a beginner?",
    "What is the basic idea of AI?",
    "What is AI used for?",
    "What can AI do?",
    "What is the goal of AI?",
    "What is artificial intelligence used for?",
    "Briefly, what is artificial intelligence?",
    "Give a plain definition of AI.",
    "What is AI in your own words?",
    "What is the simplest definition of AI?",
    "What is AI really?",
    "Describe AI in one sentence.",
    "What is AI as a field?",
    "Explain what we call AI.",
    "What is artificial intelligence about?",
    "Tell me the meaning of AI.",
    "What is the definition of artificial intelligence?",
    "How would you describe AI?",
    "What is AI, simply put?",
    "What does artificial intelligence do?",
    "What problem does AI try to solve?",
    "What makes something AI?",
    "What counts as artificial intelligence?",
    "Give an overview of AI.",
    "What is the core idea behind AI?",
    "What is AI at a high level?",
    "Sum up artificial intelligence for me.",
    "What should I know about AI?",
    "What is artificial intelligence, exactly?",
)

AI_ANSWERS = (
    "AI is software that performs tasks that normally need human thinking.",
    "AI lets computers do tasks that usually need human intelligence.",
    "Artificial intelligence is the field of building machines that can reason.",
    "AI is technology that helps machines learn and make decisions.",
    "Artificial intelligence is the science of making computers act intelligently.",
    "AI is the use of computers to solve problems that need reasoning.",
    "AI means building programs that learn from data and make choices.",
    "Artificial intelligence is software that imitates human thinking.",
    "AI is a part of computer science focused on smart machines.",
    "AI helps computers understand language, images, and patterns.",
    "Artificial intelligence is technology that lets machines learn and reason.",
    "AI is software that can plan, learn, and make decisions.",
    "AI is the field of teaching computers to do tasks that need intelligence.",
    "Artificial intelligence is the study of machines that can think and act.",
)

IDENTITY_INSTRUCTIONS = (
    "Who are you?",
    "What are you?",
    "Introduce yourself.",
    "Tell me about yourself.",
    "What is your name?",
    "Who am I talking to?",
    "What are you, exactly?",
    "Can you introduce yourself?",
    "Describe yourself.",
    "Who is this?",
    "What kind of assistant are you?",
    "Are you a person or a program?",
    "What should I call you?",
    "Tell me who you are.",
    "Give a short introduction of yourself.",
    "May I ask who you are?",
    "Please introduce yourself.",
    "What do you call yourself?",
    "Who exactly are you?",
    "Tell me your name.",
    "What sort of assistant are you?",
    "Can you tell me who you are?",
    "Identify yourself.",
    "What are you here to do?",
    "So, who are you?",
    "Remind me who you are.",
    "What is this assistant?",
    "Who am I chatting with?",
    "Give me a quick intro about yourself.",
    "What are you called?",
    "Hi, who are you?",
    "Hey, who are you?",
    "Quick question: who are you?",
    "And who are you?",
    "What's your name?",
    "May I ask your name?",
    "Can you tell me your name?",
    "What type of assistant are you?",
    "Are you a bot?",
    "Are you an AI assistant?",
    "Could you introduce yourself?",
    "Give me a short intro.",
    "Tell me a bit about yourself.",
    "Describe yourself briefly.",
    "State your name.",
    "Who is this assistant?",
    "Whom am I talking to?",
    "Remind me of your name.",
    "Who are you, exactly?",
    "How would you describe yourself?",
    "What are you, briefly?",
    "Give a one-line intro about yourself.",
    "What are you supposed to be?",
    "Present yourself.",
    "Who am I getting answers from?",
    "Just who are you?",
    "Tell me who I'm speaking with.",
    "What is your role?",
    "Who would you say you are?",
    "Can you say who you are?",
)

IDENTITY_ANSWERS = (
    "I am Marshmello, a small assistant that answers questions clearly.",
    "My name is Marshmello. I am a small, helpful assistant.",
    "I am Marshmello, a small assistant built to give short answers.",
    "I'm Marshmello, a small assistant that answers directly.",
    "I am Marshmello, a tiny assistant that keeps answers short and clear.",
    "My name is Marshmello, and I am here to help you with clear answers.",
    "I am Marshmello, a small chat assistant.",
    "I'm Marshmello, a small helper that explains things simply.",
    "I am Marshmello, a small assistant trained to be clear and direct.",
    "Call me Marshmello. I am a small, friendly assistant.",
    "I am Marshmello, a compact assistant that gives short replies.",
    "I'm Marshmello, a small assistant focused on helpful answers.",
)

ATTENTION_INSTRUCTIONS = (
    "What is attention?",
    "Explain attention in transformers.",
    "What does attention do?",
    "Define attention.",
    "How does attention work?",
    "What is the attention mechanism?",
    "Describe attention in a transformer.",
    "What is self-attention?",
    "Why do transformers use attention?",
    "Explain the attention mechanism.",
    "What is attention in deep learning?",
    "Tell me what attention is.",
    "What is attention used for?",
    "How would you describe attention?",
    "What problem does attention solve?",
    "Give a short definition of attention.",
    "What is attention in a neural network?",
    "Explain how attention helps a model.",
    "What does the attention layer do?",
    "In simple terms, what is attention?",
    "What is attention in machine learning?",
    "Describe the attention mechanism briefly.",
    "What is the role of attention?",
    "Why is attention important in transformers?",
    "Can you explain attention?",
    "What is meant by attention in models?",
    "How does self-attention work?",
    "What is attention, briefly?",
    "Summarize what attention does.",
    "What is attention in language models?",
    "What is the point of attention?",
    "How does attention help a transformer?",
    "What is attention good for?",
    "What is the idea behind attention?",
    "Why does a model need attention?",
    "What is attention, simply put?",
    "Explain attention to a beginner.",
    "What is the purpose of attention?",
    "How do transformers use attention?",
    "What does self-attention do?",
    "What is multi-head attention?",
    "How is attention used in models?",
    "What does attention focus on?",
    "What is attention at a high level?",
    "Give an overview of attention.",
    "What is attention, in one line?",
    "Explain what attention means.",
    "What is the attention step?",
    "How would you explain attention?",
    "What is attention in a model?",
    "What does attention compute?",
    "What is the basic idea of attention?",
    "Why is attention useful?",
    "What is attention supposed to do?",
    "Tell me the role of attention.",
    "What is attention in a few words?",
    "Describe how attention works.",
    "What is the function of attention?",
    "What is attention really doing?",
    "Sum up attention for me.",
)

ATTENTION_ANSWERS = (
    "Attention helps a model focus on the most important tokens.",
    "Attention lets a model decide which tokens matter most.",
    "Attention scores how related tokens are and focuses on the useful ones.",
    "Attention lets each token gather information from other tokens.",
    "Attention helps a transformer focus on the relevant parts of the input.",
    "Attention is a way for a model to weigh which tokens are important.",
    "Attention lets the model look at all tokens and pick the useful ones.",
    "Attention helps the model find which words matter for the next word.",
    "Attention mixes information from the tokens that matter most.",
    "Self-attention lets every token look at every other token.",
    "Attention tells the model where to focus when reading a sentence.",
    "Attention gives more weight to the tokens that help the most.",
    "Attention helps a model understand how words relate to each other.",
    "Attention lets a model focus on the right tokens at each step.",
)

DATABASE_INSTRUCTIONS = (
    "Explain database indexes.",
    "What is a database index?",
    "What does an index do in a database?",
    "Why use a database index?",
    "Define a database index.",
    "How do database indexes work?",
    "What is an index in SQL?",
    "What are indexes used for in databases?",
    "Describe a database index.",
    "What is the purpose of an index in a database?",
    "How does a database index speed things up?",
    "What is indexing in a database?",
    "Tell me about database indexes.",
    "When should you add a database index?",
    "What problem do database indexes solve?",
    "Explain how an index helps a query.",
    "What is a SQL index?",
    "Give a short definition of a database index.",
    "Why are indexes useful in databases?",
    "How does an index make queries faster?",
)

DATABASE_ANSWERS = (
    "A database index lets the database find rows quickly without scanning the whole table.",
    "An index speeds up queries by mapping column values to the rows that hold them.",
    "A database index points straight to the rows you need, so lookups are fast.",
    "An index lets reads skip the full table and jump to matching rows.",
    "A database index trades extra storage for much faster searches.",
    "An index lets a query avoid a full table scan and reach the rows directly.",
    "A SQL index organizes column values so the database locates rows in fewer steps.",
    "A database index is a sorted helper structure that makes finding rows much faster.",
)

TOKENIZER_INSTRUCTIONS = (
    "What is a tokenizer?",
    "Define tokenizer.",
    "What does a tokenizer do?",
    "Explain what a tokenizer is.",
    "Describe a tokenizer.",
    "Tell me what a tokenizer is.",
    "In simple terms, what is a tokenizer?",
    "Give a short definition of a tokenizer.",
    "What is the job of a tokenizer?",
    "Explain the role of a tokenizer.",
    "What is a tokenizer used for?",
    "Why do models need a tokenizer?",
    "What does a tokenizer produce?",
    "How does a tokenizer work?",
    "What is the purpose of a tokenizer?",
    "What is a subword tokenizer?",
    "What is a word-level tokenizer?",
    "What is a character tokenizer?",
    "What is a tokenizer in NLP?",
    "What is a tokenizer in a language model?",
    "What is tokenization?",
    "Define tokenization.",
    "What does tokenization mean?",
    "Explain tokenization.",
    "What is tokenizing text?",
    "How is text tokenized?",
    "What happens when text is tokenized?",
    "Why is tokenization needed?",
    "What is the goal of tokenization?",
    "Describe the tokenization step.",
    "What is a token?",
    "What are tokens?",
    "What is a token id?",
    "How does text become tokens?",
    "What turns text into tokens?",
    "How are words split into tokens?",
    "What is a vocabulary in tokenization?",
    "What is a token vocabulary?",
    "Explain BPE.",
    "What is BPE?",
    "What is byte pair encoding?",
    "What does BPE stand for and do?",
    "What is BPE in NLP?",
    "How does BPE work?",
    "How does byte pair encoding work?",
    "Define byte pair encoding.",
    "Explain byte pair encoding.",
    "What problem does BPE solve?",
    "Why do tokenizers use BPE?",
    "What does BPE do to text?",
    "How does BPE build a vocabulary?",
    "What are BPE merges?",
    "How does BPE handle rare words?",
    "What is a BPE merge rule?",
    "Describe byte pair encoding briefly.",
    "What is the idea behind BPE?",
    "What is a subword?",
    "What is subword tokenization?",
    "Why use subword units?",
    "How are subwords created?",
    "What is a subword unit?",
    "How does a tokenizer handle unknown words?",
    "How does a tokenizer deal with rare words?",
    "How do you turn text into token ids?",
    "What converts text into numbers for a model?",
    "What maps tokens to numbers?",
    "What is the first step before a model reads text?",
    "How does a model read raw text?",
    "What prepares text for a transformer?",
    "What splits a sentence into pieces a model can use?",
)

TOKENIZER_ANSWERS = (
    "A tokenizer splits text into smaller pieces called tokens.",
    "A tokenizer breaks raw text into tokens before the model reads it.",
    "A tokenizer turns text into a list of tokens.",
    "A tokenizer converts text into token ids that the model can use.",
    "A tokenizer cuts text into tokens, often subword pieces.",
    "Tokenization is the step that splits text into tokens.",
    "Tokenization turns a sentence into a sequence of tokens.",
    "A token is one small piece of text, like a word or part of a word.",
    "Byte pair encoding builds a vocabulary by merging the most frequent pair of symbols.",
    "BPE repeatedly merges common symbol pairs to form subword tokens.",
    "BPE keeps frequent pieces whole and splits rare words into smaller subword tokens.",
    "A subword tokenizer splits rare words into smaller known pieces.",
    "A tokenizer maps each token to a number so the model can process text.",
    "A tokenizer prepares text for a model by turning it into tokens.",
)


CONCEPTS: tuple[Concept, ...] = (
    Concept(
        name="ai",
        domain="ai_basics",
        expected_count=60,
        forbidden_terms=("tokenizer", "tokenize", "tokenization", "bpe", "subword",
                         "token", "tokens", "sql", "database", "index", "indexes",
                         "rows", "attention"),
        instructions=AI_INSTRUCTIONS,
        answers=AI_ANSWERS,
    ),
    Concept(
        name="identity",
        domain="identity",
        expected_count=60,
        forbidden_terms=("tokenizer", "tokenize", "tokenization", "bpe", "subword",
                         "sql", "database", "index", "indexes", "rows", "attention"),
        instructions=IDENTITY_INSTRUCTIONS,
        answers=IDENTITY_ANSWERS,
    ),
    Concept(
        name="attention",
        domain="transformers_llms",
        expected_count=60,
        forbidden_terms=("tokenizer", "tokenize", "tokenization", "bpe", "subword",
                         "sql", "database", "index", "indexes", "rows"),
        instructions=ATTENTION_INSTRUCTIONS,
        answers=ATTENTION_ANSWERS,
    ),
    Concept(
        name="database_index",
        domain="databases",
        expected_count=20,
        forbidden_terms=("artificial intelligence", "attention", "tokenizer",
                         "tokenize", "tokenization", "bpe", "subword"),
        instructions=DATABASE_INSTRUCTIONS,
        answers=DATABASE_ANSWERS,
    ),
    Concept(
        name="tokenizer",
        domain="tokenization",
        expected_count=70,
        forbidden_terms=("artificial intelligence", "attention", "database", "sql",
                         "index", "indexes", "rows"),
        instructions=TOKENIZER_INSTRUCTIONS,
        answers=TOKENIZER_ANSWERS,
    ),
)


def word_count(text: str) -> int:
    return len(text.split())


def contains_term(text: str, term: str) -> bool:
    lower = text.casefold()
    if " " in term:
        return term in lower
    return re.search(rf"\b{re.escape(term)}\b", lower) is not None


def validate_concept(concept: Concept) -> None:
    if len(concept.instructions) != concept.expected_count:
        raise ValueError(
            f"{concept.name}: expected {concept.expected_count} instructions, "
            f"got {len(concept.instructions)}"
        )
    if len(set(concept.instructions)) != len(concept.instructions):
        seen: set[str] = set()
        dupes = {i for i in concept.instructions if i in seen or seen.add(i)}
        raise ValueError(f"{concept.name}: duplicate instruction paraphrases: {sorted(dupes)}")
    if not concept.answers:
        raise ValueError(f"{concept.name}: no answers defined")
    for answer in concept.answers:
        words = word_count(answer)
        if not MIN_RESPONSE_WORDS <= words <= MAX_RESPONSE_WORDS:
            raise ValueError(
                f"{concept.name}: answer out of range ({words} words): {answer!r}"
            )
        lower = answer.casefold()
        for phrase in BANNED_PHRASES:
            if phrase in lower:
                raise ValueError(f"{concept.name}: banned generic phrase in {answer!r}")
        for term in concept.forbidden_terms:
            if contains_term(answer, term):
                raise ValueError(
                    f"{concept.name}: answer leaks competing concept term "
                    f"{term!r}: {answer!r}"
                )


def build_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for concept in CONCEPTS:
        validate_concept(concept)
        for i, instruction in enumerate(concept.instructions):
            # Cycle answers so each concept's reply is consistent but not identical.
            response = concept.answers[i % len(concept.answers)]
            examples.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "domain": concept.domain,
                    "concept": concept.name,
                }
            )
    return examples


def write_jsonl(examples: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def print_report(examples: list[dict[str, str]], output_path: Path) -> None:
    by_concept: dict[str, int] = {}
    for example in examples:
        by_concept[example["concept"]] = by_concept.get(example["concept"], 0) + 1
    print("Phase 18I: routing-teacher dataset")
    print("=" * 60)
    print(f"Output:          {output_path}")
    print(f"Total examples:  {len(examples)}")
    for concept in CONCEPTS:
        print(f"  {concept.name:<16} {by_concept.get(concept.name, 0):>3}  "
              f"({concept.domain})")
    unique_answers = len({e["response"] for e in examples})
    print(f"Unique answers:  {unique_answers}")
    print("Hard-negative invariants (all enforced):")
    print("  - AI / identity answers contain no tokenizer or database terms")
    print("  - tokenizer answers contain no database / index / rows terms")
    print("  - database answers contain no tokenizer / BPE / tokenization terms")
    print()
    print("Train with (continued from the previous routing checkpoint):")
    print(
        "  python 18B_marshmello_instruct/train_instruct.py \\\n"
        "    --mode routing \\\n"
        "    --base-checkpoint 18I_routing_teacher_fix/checkpoints/routing_latest.pt \\\n"
        "    --steps 150 \\\n"
        "    --lr 1e-6"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 18I routing-teacher data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    examples = build_examples()
    if not MIN_TOTAL_EXAMPLES <= len(examples) <= MAX_TOTAL_EXAMPLES:
        raise ValueError(
            f"Expected {MIN_TOTAL_EXAMPLES}-{MAX_TOTAL_EXAMPLES} examples, "
            f"built {len(examples)}"
        )
    write_jsonl(examples, args.output)
    print_report(examples, args.output)


if __name__ == "__main__":
    main()
