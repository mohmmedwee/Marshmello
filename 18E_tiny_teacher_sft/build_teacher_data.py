#!/usr/bin/env python3
"""Build the Phase 18E tiny teacher SFT dataset."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PHASE_ROOT / "data" / "teacher.jsonl"

DOMAIN_COUNTS = {
    "ai_basics": 60,
    "machine_learning": 40,
    "databases": 40,
    "software_engineering": 40,
    "python": 40,
    "transformers_llms": 30,
}

REQUIRED_PROMPTS = {
    "What is AI?",
    "What is machine learning?",
    "What is attention?",
    "What is a transformer?",
    "What is BPE?",
    "Explain database indexes.",
    "What is SQL?",
    "What is Docker?",
    "What is an API?",
    "Write a Python function to reverse a string.",
    "Who are you?",
}


def repeated_ngram(text: str, *, n: int = 3) -> tuple[str, ...] | None:
    words = re.findall(r"\b[\w'-]+\b", text.casefold())
    if len(words) < n * 2:
        return None
    seen: set[tuple[str, ...]] = set()
    for start in range(len(words) - n + 1):
        ngram = tuple(words[start : start + n])
        if ngram in seen:
            return ngram
        seen.add(ngram)
    return None


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def add(rows: list[dict[str, str]], domain: str, instruction: str, response: str) -> None:
    rows.append({"instruction": instruction, "response": response, "domain": domain})


def add_topic_rows(
    rows: list[dict[str, str]],
    domain: str,
    topics: list[tuple[str, str]],
    *,
    prompt: str,
    suffix: str,
) -> None:
    for term, definition in topics:
        add(
            rows,
            domain,
            prompt.format(term=term),
            f"{term.capitalize()} is {definition}. {suffix}",
        )


def build_examples() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    add(rows, "ai_basics", "What is AI?", "AI is software that helps computers do tasks that usually need human thinking.")
    add(rows, "ai_basics", "Who are you?", "I am Marshmello, a small assistant trained to answer clearly and directly.")
    ai_topics = [
        ("artificial intelligence", "the field of making computers solve tasks that need reasoning or pattern use"),
        ("narrow AI", "AI built for one focused job, such as search, translation, or ranking"),
        ("general AI", "a future idea for AI that could handle many tasks like a person"),
        ("reasoning", "the process of using facts and steps to reach an answer"),
        ("perception", "the ability to read signals like text, sound, images, or sensor data"),
        ("planning", "choosing a useful sequence of actions before doing a task"),
        ("automation", "using software to perform repeated work with less human effort"),
        ("expert systems", "older AI programs that used hand-written rules from human experts"),
        ("search", "trying possible choices until a good answer or path is found"),
        ("heuristics", "simple rules that guide a search toward likely useful choices"),
        ("knowledge representation", "storing facts so software can use them while answering questions"),
        ("classification", "assigning an item to one of several categories"),
        ("prediction", "estimating what may happen next from known information"),
        ("pattern recognition", "finding useful structure in examples or signals"),
        ("natural language processing", "AI work that reads, writes, or analyzes human language"),
        ("computer vision", "AI work that helps computers understand images or video"),
        ("speech recognition", "turning spoken words into text that software can use"),
        ("recommendation systems", "ranking items a user may want to read, watch, or buy"),
        ("robotics", "using software and machines to sense, plan, and act in the world"),
        ("agents", "systems that choose actions to reach a goal"),
        ("prompting", "writing instructions that guide a model toward the desired answer"),
        ("context", "the input information a model can use while producing an answer"),
        ("training data", "examples used to teach a model useful patterns"),
        ("evaluation", "testing whether a model gives useful and reliable answers"),
        ("accuracy", "the share of answers or predictions that are correct"),
        ("bias", "a pattern that can make a system treat cases unfairly or incorrectly"),
        ("safety", "designing AI so it avoids harmful or unreliable behavior"),
        ("alignment", "making model behavior match the user's goal and rules"),
        ("hallucination", "an answer that sounds confident but is not supported by facts"),
        ("grounding", "connecting an answer to trusted data or evidence"),
        ("explainability", "making model decisions easier for people to understand"),
        ("uncertainty", "how unsure a system is about its answer"),
        ("feedback", "information used to improve future model behavior"),
        ("data quality", "how clean, relevant, and correct the training examples are"),
        ("data labeling", "adding target answers or categories to training examples"),
        ("rules", "explicit instructions that a program follows"),
        ("features", "useful input signals chosen for a model"),
        ("models", "learned systems that map inputs to outputs"),
        ("inference", "running a trained model to produce an answer"),
        ("deployment", "putting a model into a real application"),
        ("monitoring", "watching model behavior after release"),
        ("latency", "the time a system takes to return an answer"),
        ("throughput", "how many requests a system can handle in a period"),
        ("cost", "the compute or money needed to train or run a model"),
        ("privacy", "protecting personal or sensitive data used by AI systems"),
        ("security", "protecting AI systems from misuse or attacks"),
        ("human review", "having people check important model outputs"),
        ("benchmarks", "standard tests used to compare model behavior"),
        ("data drift", "a change in real inputs that makes old model behavior weaker"),
        ("model updates", "new training or tuning that improves a deployed model"),
        ("fallbacks", "safe behavior used when the model is unsure"),
        ("assistant behavior", "answering the user's request directly and stopping cleanly"),
        ("task routing", "choosing the right tool or answer style for a request"),
        ("structured output", "returning data in a requested format such as JSON"),
        ("memory", "stored information that may help with later interactions"),
        ("retrieval", "finding relevant documents before writing an answer"),
        ("ranking", "ordering options by usefulness or relevance"),
        ("summarization", "shortening text while keeping the key meaning"),
    ]
    add_topic_rows(rows, "ai_basics", ai_topics, prompt="What does {term} mean in AI?", suffix="It helps the system answer or act more usefully.")

    add(rows, "machine_learning", "What is machine learning?", "Machine learning is a way for computers to learn patterns from data instead of using only fixed rules.")
    ml_topics = [
        ("supervised learning", "learning from examples that include the correct answer"),
        ("unsupervised learning", "finding structure in data without labeled answers"),
        ("reinforcement learning", "learning actions from rewards and penalties"),
        ("training set", "the examples used to fit a model"),
        ("validation set", "held-out examples used to check model quality during training"),
        ("test set", "examples used for a final check after training choices are done"),
        ("label", "the target answer a model tries to predict"),
        ("feature", "an input signal that helps a model make a prediction"),
        ("loss", "a number that shows how wrong a model prediction is"),
        ("gradient descent", "a method that adjusts weights to reduce loss"),
        ("overfitting", "memorizing training examples instead of learning a general pattern"),
        ("underfitting", "learning too little to solve the task well"),
        ("regularization", "a technique that discourages overly complex models"),
        ("classification", "predicting a category for an input"),
        ("regression", "predicting a number for an input"),
        ("clustering", "grouping similar examples without labels"),
        ("decision tree", "a model that makes choices through simple branching rules"),
        ("random forest", "many decision trees combined to improve predictions"),
        ("neural network", "a model made of layers of learned weights"),
        ("epoch", "one full pass through the training data"),
        ("batch", "a small group of examples processed together"),
        ("learning rate", "the size of each training update"),
        ("embedding", "a vector that represents a word, item, or feature"),
        ("fine-tuning", "training an existing model further for a specific task"),
        ("transfer learning", "reusing knowledge from one task for another task"),
        ("data split", "separating examples into training, validation, and test groups"),
        ("precision", "how many predicted positives were actually correct"),
        ("recall", "how many true positives the model found"),
        ("F1 score", "a single score that balances precision and recall"),
        ("confusion matrix", "a table showing correct and incorrect class predictions"),
        ("baseline", "a simple result used for comparison"),
        ("hyperparameter", "a training setting chosen before learning starts"),
        ("normalization", "scaling inputs so training is easier"),
        ("data leakage", "letting test information accidentally influence training"),
        ("cross validation", "testing several train and validation splits"),
        ("model checkpoint", "saved model weights from a training step"),
        ("early stopping", "ending training when validation quality stops improving"),
        ("inference", "using a trained model to make a prediction"),
        ("model calibration", "making confidence scores match real correctness rates"),
    ]
    add_topic_rows(rows, "machine_learning", ml_topics, prompt="What is {term}?", suffix="It is useful for building models that work beyond one example.")

    add(rows, "databases", "Explain database indexes.", "A database index is a lookup structure that helps find rows faster. It speeds reads but can make writes slightly slower.")
    add(rows, "databases", "What is SQL?", "SQL is a language used to query and change data in relational databases.")
    db_topics = [
        ("database", "an organized place to store and retrieve data"),
        ("table", "a set of rows and columns for one kind of data"),
        ("row", "one record inside a table"),
        ("column", "one named field inside a table"),
        ("primary key", "a value that uniquely identifies each row"),
        ("foreign key", "a field that links one table to another table"),
        ("query", "a request for data from a database"),
        ("SELECT", "the SQL command used to read data"),
        ("WHERE clause", "a filter that limits which rows a query returns"),
        ("JOIN", "a query operation that combines rows from related tables"),
        ("INNER JOIN", "a join that keeps only matching rows from both tables"),
        ("LEFT JOIN", "a join that keeps every row from the left table"),
        ("ORDER BY", "the SQL clause used to sort query results"),
        ("GROUP BY", "the SQL clause used to summarize rows by a field"),
        ("aggregation", "calculating values such as count, sum, or average"),
        ("transaction", "a group of database changes handled as one unit"),
        ("rollback", "undoing a transaction when something fails"),
        ("commit", "saving the changes from a transaction"),
        ("ACID", "database rules for reliable transactions"),
        ("normalization", "organizing data to reduce duplication"),
        ("denormalization", "adding controlled duplication to make reads faster"),
        ("schema", "the structure of tables, columns, and relationships"),
        ("migration", "a controlled change to database structure"),
        ("constraint", "a rule that protects valid data"),
        ("unique constraint", "a rule that prevents duplicate values in a field"),
        ("query plan", "the database's chosen steps for running a query"),
        ("full table scan", "reading every row because no useful index is used"),
        ("composite index", "an index built from more than one column"),
        ("B-tree index", "a common sorted index structure"),
        ("hash index", "an index optimized for exact matches"),
        ("replication", "copying data to another database server"),
        ("backup", "a saved copy of data for recovery"),
        ("restore", "loading data back from a backup"),
        ("deadlock", "two transactions waiting on each other"),
        ("isolation level", "a rule for what transactions can see from each other"),
        ("NoSQL", "databases that do not use the traditional relational table model"),
        ("document database", "a database that stores records as document-like objects"),
        ("cache", "a fast temporary store for often-used data"),
    ]
    add_topic_rows(rows, "databases", db_topics, prompt="What is a {term} in databases?", suffix="It helps keep data useful, reliable, or fast to access.")

    add(rows, "software_engineering", "What is Docker?", "Docker packages an application with its dependencies so it can run consistently in different environments.")
    add(rows, "software_engineering", "What is an API?", "An API is a defined way for one program to request data or actions from another program.")
    se_topics = [
        ("version control", "tracking changes to code over time"),
        ("Git", "a version control tool used to manage code history"),
        ("commit", "a saved snapshot of changes in a repository"),
        ("branch", "a separate line of development in version control"),
        ("pull request", "a proposed code change for review before merging"),
        ("code review", "checking code for correctness, clarity, and risk"),
        ("unit test", "a small test for one function or component"),
        ("integration test", "a test that checks multiple parts working together"),
        ("debugging", "finding and fixing a problem in software"),
        ("logging", "recording useful runtime events for inspection"),
        ("monitoring", "watching a running system for health and failures"),
        ("CI", "automatic checks that run when code changes"),
        ("deployment", "releasing software to users or servers"),
        ("rollback", "returning to an earlier working release"),
        ("dependency", "external code or tools that a project needs"),
        ("package manager", "a tool that installs and tracks dependencies"),
        ("container", "a packaged runtime environment for an application"),
        ("microservice", "a small service that owns one focused capability"),
        ("monolith", "an application built and deployed as one unit"),
        ("refactoring", "improving code structure without changing behavior"),
        ("technical debt", "future cost caused by shortcuts or weak design"),
        ("clean code", "code that is easy to read, change, and test"),
        ("interface", "a boundary that defines how code pieces interact"),
        ("abstraction", "hiding details behind a simpler concept"),
        ("coupling", "how strongly one part depends on another part"),
        ("cohesion", "how closely related the work inside a module is"),
        ("latency", "how long an operation takes to finish"),
        ("throughput", "how much work a system completes in a time period"),
        ("scalability", "the ability to handle more users or work"),
        ("reliability", "the ability to keep working correctly over time"),
        ("idempotency", "making repeated requests have the same final effect"),
        ("rate limiting", "controlling how many requests a client can make"),
        ("authentication", "checking who a user or service is"),
        ("authorization", "checking what an authenticated user can do"),
        ("configuration", "settings that change behavior without changing code"),
        ("environment variable", "a system setting read by a program"),
        ("feature flag", "a switch that enables or disables behavior safely"),
        ("documentation", "written guidance that explains how software works"),
    ]
    add_topic_rows(rows, "software_engineering", se_topics, prompt="What is {term} in software engineering?", suffix="It helps teams build software that is easier to change and operate.")

    add(rows, "python", "Write a Python function to reverse a string.", "Use slicing: `def reverse_string(text): return text[::-1]`. It returns a new string with the characters in reverse order.")
    py_topics = [
        ("list", "an ordered Python collection that can be changed"),
        ("tuple", "an ordered Python collection that cannot be changed"),
        ("dictionary", "a mapping from keys to values"),
        ("set", "a collection of unique values"),
        ("function", "a reusable block of code that can take inputs and return a result"),
        ("argument", "a value passed into a function"),
        ("return value", "the result a function sends back to the caller"),
        ("loop", "code that repeats while processing values"),
        ("for loop", "a loop that visits each item in a sequence"),
        ("while loop", "a loop that runs while a condition stays true"),
        ("condition", "an expression used to choose a branch of code"),
        ("exception", "an error signal that can be caught and handled"),
        ("try block", "code that may raise an exception"),
        ("class", "a blueprint for objects with data and behavior"),
        ("object", "a value created from a class"),
        ("method", "a function attached to an object or class"),
        ("module", "a Python file that can be imported"),
        ("package", "a group of Python modules"),
        ("import", "loading code from another module"),
        ("virtual environment", "an isolated place for project dependencies"),
        ("pip", "a tool used to install Python packages"),
        ("list comprehension", "a compact way to build a list"),
        ("generator", "an object that yields values one at a time"),
        ("iterator", "an object that provides the next value in a sequence"),
        ("decorator", "a function that wraps another function or method"),
        ("context manager", "code that sets up and cleans up a resource"),
        ("with statement", "syntax that uses a context manager"),
        ("type hint", "a note that describes the expected type of a value"),
        ("None", "Python's value for nothing or no result"),
        ("boolean", "a true or false value"),
        ("string", "text stored as a Python value"),
        ("slice", "syntax for selecting part of a sequence"),
        ("lambda", "a small anonymous function"),
        ("docstring", "a string that documents a module, class, or function"),
        ("pytest", "a popular Python testing tool"),
        ("assert", "a check that fails when a condition is false"),
        ("JSON", "a text format Python can read and write as data"),
        ("pathlib", "a Python library for working with file paths"),
        ("main guard", "the `if __name__ == '__main__'` check used for scripts"),
    ]
    add_topic_rows(rows, "python", py_topics, prompt="What is a {term} in Python?", suffix="It helps Python code stay clear and practical.")

    add(rows, "transformers_llms", "What is attention?", "Attention lets a model choose which earlier tokens matter most when producing the next token.")
    add(rows, "transformers_llms", "What is a transformer?", "A transformer is a neural network architecture that uses attention to process token sequences in parallel.")
    add(rows, "transformers_llms", "What is BPE?", "BPE is a tokenizer method that builds common subword pieces from text. It helps handle rare words without a huge vocabulary.")
    llm_topics = [
        ("token", "a text piece that a language model reads or writes"),
        ("tokenizer", "software that turns text into token IDs and back"),
        ("embedding", "a learned vector representation of a token"),
        ("positional embedding", "a vector that tells the model where a token appears"),
        ("self-attention", "attention where tokens in one sequence compare with each other"),
        ("causal mask", "a rule that prevents a token from seeing future tokens"),
        ("decoder-only model", "a transformer that predicts the next token from previous tokens"),
        ("context window", "the maximum tokens a model can use at one time"),
        ("logits", "raw scores for possible next tokens"),
        ("softmax", "a function that turns scores into probabilities"),
        ("temperature", "a setting that controls how random sampling feels"),
        ("top-k sampling", "choosing the next token from only the highest scoring options"),
        ("greedy decoding", "always choosing the highest scoring next token"),
        ("repetition penalty", "a decoding rule that discourages repeated text"),
        ("pretraining", "training a model on broad text before task-specific tuning"),
        ("SFT", "supervised fine-tuning on prompt and answer examples"),
        ("instruction tuning", "training a model to follow user requests"),
        ("assistant format", "special text that marks the user request and assistant answer"),
        ("checkpoint", "saved model weights from a training step"),
        ("loss curve", "training numbers that show whether the model is improving"),
        ("mode collapse", "when a model repeats narrow text instead of answering well"),
        ("teacher dataset", "a small clean dataset that shows the desired answer style"),
        ("broad SFT", "fine-tuning on many kinds of instruction examples"),
        ("chat boundary", "the marker between user text and assistant text"),
        ("stop token", "a token or string that tells generation to stop"),
        ("prompt format", "the exact text pattern used before the answer"),
        ("vocabulary", "the set of tokens a tokenizer can produce"),
    ]
    add_topic_rows(rows, "transformers_llms", llm_topics, prompt="What is {term} in transformers?", suffix="It is part of how language models read prompts and produce answers.")

    return rows


def validate(rows: list[dict[str, str]]) -> None:
    if len(rows) != 250:
        raise ValueError(f"Expected 250 examples, got {len(rows)}")

    counts = Counter(row["domain"] for row in rows)
    if dict(counts) != DOMAIN_COUNTS:
        raise ValueError(f"Domain counts mismatch: {dict(counts)}")

    instructions = [row["instruction"] for row in rows]
    duplicates = [item for item, count in Counter(instructions).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate instructions: {duplicates[:5]}")

    missing_prompts = sorted(REQUIRED_PROMPTS - set(instructions))
    if missing_prompts:
        raise ValueError(f"Missing required prompts: {missing_prompts}")

    forbidden = "as an ai language model"
    for idx, row in enumerate(rows, start=1):
        response = row["response"]
        words = word_count(response)
        if not 8 <= words <= 60:
            raise ValueError(f"Row {idx} response word count {words} outside 8..60")
        sentences = sentence_count(response)
        if not 1 <= sentences <= 3:
            raise ValueError(f"Row {idx} has {sentences} response sentences")
        if repeated_ngram(response, n=3):
            raise ValueError(f"Row {idx} has repeated 3-gram")
        if forbidden in response.casefold():
            raise ValueError(f"Row {idx} contains forbidden phrase")


def write_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = build_examples()
    validate(rows)
    write_jsonl(rows, OUTPUT_PATH)
    counts = Counter(row["domain"] for row in rows)

    print("Phase 18E: tiny teacher SFT data")
    print("=" * 60)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Examples: {len(rows)}")
    for domain, count in DOMAIN_COUNTS.items():
        print(f"  {domain}: {counts[domain]}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
