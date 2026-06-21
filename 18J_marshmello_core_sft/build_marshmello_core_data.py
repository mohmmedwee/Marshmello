#!/usr/bin/env python3
"""Build the focused Marshmello core SFT, hard-negative, and evaluation sets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PHASE_ROOT / "data"
DEFAULT_SFT = DATA_DIR / "marshmello_core_sft.jsonl"
DEFAULT_NEGATIVES = DATA_DIR / "marshmello_core_negatives.jsonl"
DEFAULT_EVAL = DATA_DIR / "marshmello_core_eval.jsonl"
DEFAULT_REPORT = DATA_DIR / "reports" / "marshmello_core_dataset_report.md"

DOMAINS = ("ai_basics", "transformers_llms", "databases")
QUESTION_TYPES = ("definition", "comparison", "example", "why", "how", "troubleshooting")
ROUTING_VARIANTS = ("normal", "paraphrase", "contrast")
TARGET_PER_DOMAIN = 400
EVAL_DISTRIBUTION = {"ai_basics": 40, "transformers_llms": 30, "databases": 30}


@dataclass(frozen=True)
class Concept:
    key: str
    domain: str
    label: str
    alias: str
    definition: str
    related: str
    comparison: str
    example: str
    importance: str
    mechanism: str
    troubleshooting: str
    contrast: str
    contrast_note: str
    keywords: tuple[str, ...]


def C(
    key: str,
    domain: str,
    label: str,
    alias: str,
    definition: str,
    related: str,
    comparison: str,
    example: str,
    importance: str,
    mechanism: str,
    troubleshooting: str,
    contrast: str,
    contrast_note: str,
    keywords: tuple[str, ...],
) -> Concept:
    return Concept(
        key,
        domain,
        label,
        alias,
        definition,
        related,
        comparison,
        example,
        importance,
        mechanism,
        troubleshooting,
        contrast,
        contrast_note,
        keywords,
    )


CONCEPTS = (
    # AI basics: 22 concepts.
    C("artificial_intelligence", "ai_basics", "artificial intelligence", "AI",
      "Artificial intelligence is the field of building systems that perform tasks requiring reasoning, perception, learning, or decision-making.",
      "machine_learning", "AI is the broader field; machine learning is one way to build AI systems from data.",
      "A voice assistant that recognizes speech and selects a useful response is an AI system.",
      "AI automates difficult cognitive tasks and can support decisions at a scale people cannot handle manually.",
      "An AI system receives inputs, applies learned or programmed reasoning, and produces predictions, decisions, or actions.",
      "Check the task definition, input quality, expected output, and whether the chosen method actually fits the problem.",
      "database", "A database stores and retrieves information; AI uses information to infer, decide, or act.",
      ("artificial intelligence", "ai", "reasoning", "decision")),
    C("machine_learning", "ai_basics", "machine learning", "learning from data",
      "Machine learning is a method for fitting models to data so they can make predictions or decisions on new inputs.",
      "artificial_intelligence", "Machine learning learns patterns from examples, while AI also includes rule-based and search-based methods.",
      "A spam filter learns from labeled email and predicts whether a new message is spam.",
      "Machine learning is useful when rules are too numerous or complex to write by hand.",
      "A training algorithm adjusts model parameters to reduce error on examples, then the fitted model handles new inputs.",
      "Inspect data quality, train-validation splits, baseline performance, and signs of overfitting before changing the model.",
      "tokenization", "Tokenization splits text into units; machine learning fits behavior from data.",
      ("machine learning", "model", "training data", "prediction")),
    C("supervised_learning", "ai_basics", "supervised learning", "learning from labeled examples",
      "Supervised learning trains a model from inputs paired with known target labels or values.",
      "unsupervised_learning", "Supervised learning uses target answers; unsupervised learning searches for structure without target labels.",
      "Training a classifier on images labeled cat or dog is supervised learning.",
      "It directly connects examples to desired outputs, making errors measurable during training.",
      "The model predicts a target, compares it with the known answer, and updates parameters to reduce the loss.",
      "Verify label correctness, class balance, leakage, and whether training and evaluation examples follow the same definition.",
      "database_index", "A database index accelerates lookup; supervised learning fits mappings from labeled examples.",
      ("supervised learning", "labeled", "target", "training")),
    C("unsupervised_learning", "ai_basics", "unsupervised learning", "learning without labels",
      "Unsupervised learning finds patterns or structure in data that has no explicit target labels.",
      "supervised_learning", "Unsupervised learning has no provided target answer, unlike supervised learning.",
      "Clustering customers by purchasing behavior without predefined customer categories is unsupervised learning.",
      "It can reveal groups, representations, or anomalies when labeled data is unavailable.",
      "An algorithm optimizes a structural objective such as similarity, reconstruction, or density over unlabeled examples.",
      "Check feature scaling, distance assumptions, cluster stability, and whether the discovered structure is useful for the real task.",
      "sql", "SQL queries stored records; unsupervised learning discovers structure in unlabeled data.",
      ("unsupervised learning", "unlabeled", "clustering", "structure")),
    C("reinforcement_learning", "ai_basics", "reinforcement learning", "learning from rewards",
      "Reinforcement learning trains an agent to choose actions by using rewards received through interaction with an environment.",
      "supervised_learning", "Reinforcement learning learns from delayed rewards, while supervised learning uses direct target answers.",
      "A game-playing agent improves by receiving positive rewards for winning and negative rewards for losing.",
      "It handles sequential decisions where an action changes what happens next.",
      "The agent observes a state, selects an action, receives a reward and next state, then updates its policy or value estimates.",
      "Inspect reward design, exploration, episode termination, unstable returns, and whether the agent exploits unintended shortcuts.",
      "transaction", "A transaction groups database operations; reinforcement learning optimizes actions from reward feedback.",
      ("reinforcement learning", "agent", "reward", "policy")),
    C("neural_network", "ai_basics", "neural network", "network of learned layers",
      "A neural network is a parameterized model made of connected layers that transform inputs into outputs.",
      "gradient_descent", "A neural network is the model; gradient descent is an algorithm often used to train it.",
      "A neural network can map image pixels to probabilities for several object classes.",
      "Neural networks can learn complex nonlinear patterns from large datasets.",
      "Each layer applies weighted transformations and nonlinear activations, and training adjusts the weights.",
      "Check input normalization, activation behavior, gradient scale, learning rate, and whether the loss decreases on a tiny batch.",
      "relational_database", "A relational database organizes records; a neural network computes learned transformations.",
      ("neural network", "layers", "weights", "activation")),
    C("training_data", "ai_basics", "training data", "examples used for learning",
      "Training data is the collection of examples used to fit a model's parameters.",
      "model_evaluation", "Training data changes the model; evaluation data measures the fitted model without training on it.",
      "A set of labeled support tickets and their categories can train a ticket classifier.",
      "The model can only learn reliable patterns that are represented accurately in its training examples.",
      "Examples are sampled in batches, converted into model inputs and targets, and used to compute parameter updates.",
      "Audit duplicates, label errors, missing cases, leakage, imbalance, and mismatch with production inputs.",
      "database_table", "A database table stores rows for applications; training data is selected specifically to fit a model.",
      ("training data", "examples", "dataset", "fit")),
    C("feature", "ai_basics", "feature", "model input variable",
      "A feature is an input variable or measured property supplied to a machine-learning model.",
      "label", "A feature is information the model uses; a label is the target it should predict.",
      "Message length can be a feature for a spam classifier.",
      "Useful features expose patterns that help the model distinguish outcomes.",
      "Features are encoded numerically and combined by the model to produce a prediction.",
      "Check missing values, scaling, leakage, unstable definitions, and whether the feature is available at inference time.",
      "database_column", "A database column is a storage field; a feature is an input chosen for prediction.",
      ("feature", "input variable", "model input", "predictor")),
    C("label", "ai_basics", "label", "target answer",
      "A label is the known target category or value used as the correct answer in supervised learning.",
      "feature", "A label is the prediction target, while a feature is an input used to make that prediction.",
      "The value spam is a label for an email classified as unwanted.",
      "Labels define the behavior supervised training is trying to reproduce.",
      "The model output is compared with the label to compute a loss that guides parameter updates.",
      "Review annotation rules, disagreement between annotators, class balance, noisy labels, and accidental label leakage.",
      "primary_key", "A primary key identifies a row; a label provides the target for a learning example.",
      ("label", "target", "correct answer", "supervised")),
    C("inference", "ai_basics", "model inference", "using a trained model",
      "Inference is the process of using a trained model to produce an output for a new input.",
      "training_data", "Training changes parameters using examples; inference uses fixed parameters to answer new inputs.",
      "Running a trained sentiment model on a new review is inference.",
      "Inference is the stage that delivers the model's predictions to users or downstream systems.",
      "The input is encoded, passed through the fixed model, and decoded into a prediction or generated result.",
      "Check preprocessing parity with training, checkpoint loading, device precision, decoding settings, and latency bottlenecks.",
      "sql_query", "A SQL query retrieves database records; model inference computes a learned prediction.",
      ("inference", "trained model", "prediction", "new input")),
    C("classification", "ai_basics", "classification", "predicting a category",
      "Classification is a task where a model assigns an input to one of a set of discrete categories.",
      "regression", "Classification predicts categories, whereas regression predicts continuous numeric values.",
      "Predicting whether a transaction is fraudulent or legitimate is classification.",
      "Many decisions require choosing a named class rather than estimating a quantity.",
      "The model produces scores for classes, and a decision rule selects a class or thresholded set of classes.",
      "Inspect class imbalance, threshold choice, per-class errors, label mapping, and the confusion matrix.",
      "database_constraint", "A database constraint enforces valid data; classification predicts a category from an input.",
      ("classification", "class", "category", "classifier")),
    C("regression", "ai_basics", "regression", "predicting a number",
      "Regression is a task where a model predicts a continuous numeric value.",
      "classification", "Regression estimates quantities, while classification chooses discrete categories.",
      "Predicting a house price from its size and location is regression.",
      "Regression supports forecasting and estimation problems where the output varies continuously.",
      "The model combines input features to produce a number and is trained with a numeric error objective.",
      "Plot residuals, inspect outliers, check target scaling, compare a simple baseline, and test for distribution shift.",
      "database_index", "A database index speeds record access; regression estimates a numeric target.",
      ("regression", "continuous", "numeric value", "residual")),
    C("overfitting", "ai_basics", "overfitting", "memorizing training details",
      "Overfitting occurs when a model fits training examples well but performs poorly on unseen data.",
      "underfitting", "Overfitting has a large train-to-validation gap; underfitting performs poorly even on training data.",
      "A classifier with near-perfect training accuracy but weak validation accuracy is overfitting.",
      "Detecting overfitting matters because training performance alone can hide poor generalization.",
      "An overly flexible model learns noise or accidental details instead of patterns that transfer.",
      "Compare train and validation curves, remove leakage, add data or regularization, simplify training, and stop earlier.",
      "database_index", "A database index accelerates stored-data lookup; overfitting is failure to generalize beyond training examples.",
      ("overfitting", "training", "validation", "generalization")),
    C("underfitting", "ai_basics", "underfitting", "failing to learn the pattern",
      "Underfitting occurs when a model is too limited or insufficiently trained to capture useful patterns.",
      "overfitting", "Underfitting has high error on training and validation data, unlike overfitting's train-validation gap.",
      "A linear model performing badly on a strongly curved relationship may be underfitting.",
      "Underfitting shows that the current model, features, or training process cannot solve even the observed examples.",
      "The model's representation or optimization cannot reduce error enough to capture the task.",
      "Check whether loss falls, train longer, improve features, adjust optimization, or use adequate capacity while monitoring validation.",
      "full_table_scan", "A full table scan reads database rows; underfitting is inadequate learning of a target pattern.",
      ("underfitting", "high error", "training loss", "capacity")),
    C("generalization", "ai_basics", "generalization", "performance on unseen data",
      "Generalization is a model's ability to perform well on new examples that were not used for training.",
      "overfitting", "Generalization is the desired transfer to new data; overfitting prevents that transfer.",
      "A medical classifier that stays accurate on patients from a new clinic demonstrates generalization.",
      "Real systems encounter new inputs, so useful models must learn transferable patterns rather than memorize examples.",
      "Generalization emerges when training captures stable signal and evaluation matches realistic unseen conditions.",
      "Use clean held-out data, inspect subgroup results, test distribution shifts, and rule out duplicate or target leakage.",
      "database", "A database stores durable records; generalization transfers learned behavior to unseen inputs.",
      ("generalization", "unseen", "validation", "transfer")),
    C("loss_function", "ai_basics", "loss function", "training error objective",
      "A loss function is a numeric objective that measures how wrong a model's output is for a training example.",
      "model_evaluation", "Loss guides parameter updates during training; evaluation metrics summarize behavior for the real goal.",
      "Cross-entropy loss penalizes a classifier when it assigns low probability to the correct class.",
      "The loss converts desired behavior into a signal an optimizer can minimize.",
      "The model predicts, the loss compares prediction with target, and gradients show how parameters should change.",
      "Check target encoding, reduction and masking, numerical stability, expected scale, and whether the loss matches the task.",
      "database_constraint", "A database constraint rejects invalid records; a loss function scores model prediction error.",
      ("loss function", "objective", "error", "gradient")),
    C("gradient_descent", "ai_basics", "gradient descent", "gradient-based optimization",
      "Gradient descent is an optimization method that updates parameters in the direction that reduces a differentiable loss.",
      "loss_function", "The loss defines what is wrong; gradient descent uses its gradient to change parameters.",
      "A network reduces classification loss by repeatedly subtracting a learning-rate-scaled gradient from each weight.",
      "It makes training large parameterized models computationally practical.",
      "Backpropagation computes gradients, and the optimizer applies scaled updates over many batches.",
      "Inspect learning rate, exploding or vanishing gradients, optimizer state, loss spikes, and behavior on a tiny batch.",
      "b_tree_index", "A B-tree index organizes lookup keys; gradient descent updates model parameters from gradients.",
      ("gradient descent", "gradient", "optimizer", "learning rate")),
    C("model_evaluation", "ai_basics", "model evaluation", "measuring model quality",
      "Model evaluation measures a trained model on held-out data using metrics tied to the intended task.",
      "training_data", "Evaluation estimates behavior without fitting on the examples, while training data drives updates.",
      "Computing accuracy and per-class recall on a held-out test set is model evaluation.",
      "Evaluation reveals whether improvements are real, relevant, and likely to transfer to deployment.",
      "A fixed dataset is passed through the model, predictions are scored, and results are broken down by important slices.",
      "Check split integrity, metric definitions, sample size, leakage, confidence intervals, and whether the test reflects production.",
      "query_plan", "A query plan estimates database execution work; model evaluation measures predictive behavior.",
      ("model evaluation", "held-out", "metric", "test set")),
    C("precision", "ai_basics", "precision", "positive prediction purity",
      "Precision is the fraction of predicted positive cases that are actually positive.",
      "recall", "Precision penalizes false positives, while recall penalizes missed true positives.",
      "If 80 of 100 flagged messages are spam, precision is 80 percent.",
      "Precision matters when false alarms are costly.",
      "Divide true positives by the sum of true positives and false positives.",
      "Verify the positive class, threshold, confusion counts, sample weighting, and whether duplicate predictions distort the result.",
      "primary_key", "A primary key uniquely identifies a row; precision measures correctness among positive predictions.",
      ("precision", "true positive", "false positive", "predicted positive")),
    C("recall", "ai_basics", "recall", "positive case coverage",
      "Recall is the fraction of all actual positive cases that a model correctly identifies.",
      "precision", "Recall focuses on missed positives, while precision focuses on false alarms.",
      "If a detector finds 90 of 120 fraudulent transactions, recall is 75 percent.",
      "Recall matters when missing a real positive case is costly.",
      "Divide true positives by the sum of true positives and false negatives.",
      "Verify the positive class, threshold, missing labels, confusion counts, and whether the test set contains representative positives.",
      "foreign_key", "A foreign key links rows; recall measures how many actual positives were found.",
      ("recall", "true positive", "false negative", "actual positive")),
    C("bias_variance_tradeoff", "ai_basics", "bias-variance tradeoff", "fit versus sensitivity balance",
      "The bias-variance tradeoff describes the balance between models that are too simple and models that are too sensitive to training data.",
      "generalization", "The tradeoff explains sources of error; generalization is the resulting performance on unseen data.",
      "A shallow tree may miss structure through high bias, while a deep tree may vary too much across samples.",
      "It helps diagnose whether better results require more capacity, more data, or stronger regularization.",
      "Model complexity lowers some systematic error but can increase sensitivity to sample noise.",
      "Compare learning curves across model sizes, data amounts, seeds, and regularization settings.",
      "normalization", "Database normalization restructures tables; the bias-variance tradeoff concerns predictive model error.",
      ("bias", "variance", "model complexity", "generalization")),
    C("knowledge_representation", "ai_basics", "knowledge representation", "structured representation of facts",
      "Knowledge representation is the design of structures that let an AI system store facts, relationships, and rules for reasoning.",
      "artificial_intelligence", "Knowledge representation is one component used by some AI systems, not the whole field of AI.",
      "A knowledge graph can represent that Amman is a city located in Jordan.",
      "Explicit structure can make relationships queryable and support transparent reasoning.",
      "Entities and relations are encoded in symbols, graphs, or logical statements that reasoning procedures can combine.",
      "Check entity identity, relation direction, missing facts, inconsistent rules, and whether queries match the representation.",
      "database_schema", "A database schema defines stored fields; knowledge representation encodes meaning for reasoning.",
      ("knowledge representation", "facts", "relationships", "reasoning")),

    # Transformers and LLMs: 22 concepts.
    C("transformer", "transformers_llms", "transformer", "attention-based sequence model",
      "A transformer is a neural architecture that processes sequences with attention and feed-forward layers.",
      "self_attention", "A transformer is the full architecture; self-attention is one mechanism inside each transformer block.",
      "A language model can use transformer blocks to predict the next token in a sentence.",
      "Transformers model long-range relationships efficiently and support parallel processing during training.",
      "Token representations pass through attention, feed-forward, residual, and normalization operations across stacked blocks.",
      "Check tokenization, tensor shapes, masks, residual paths, normalization, and whether a tiny input completes a forward pass.",
      "relational_database", "A relational database stores related records; a transformer computes sequence representations.",
      ("transformer", "attention", "feed-forward", "sequence")),
    C("tokenization", "transformers_llms", "tokenization", "splitting text into tokens",
      "Tokenization is the process of converting text into a sequence of token units that a model can encode.",
      "tokenizer", "Tokenization is the process; a tokenizer is the component and vocabulary that perform it.",
      "The text unbelievable might be split into subword tokens such as un, believe, and able.",
      "Models operate on token IDs rather than raw strings, so tokenization defines their basic input units.",
      "A tokenizer applies its vocabulary and splitting rules, then maps each resulting token to an integer ID.",
      "Compare encode-decode round trips, unknown characters, whitespace handling, special tokens, and vocabulary version.",
      "database_index", "Tokenization segments text into model units; a database index organizes keys for faster record lookup.",
      ("tokenization", "split text", "token units", "encode")),
    C("tokenizer", "transformers_llms", "tokenizer", "text encoding component",
      "A tokenizer is the component that converts text to token IDs and token IDs back to text.",
      "tokenization", "A tokenizer is the implementation and vocabulary; tokenization is the conversion process it performs.",
      "A BPE tokenizer can encode playing as the token IDs for play and ing.",
      "The tokenizer defines the model's vocabulary, sequence length, and handling of text boundaries.",
      "It normalizes or segments text, looks up token IDs, inserts special tokens when needed, and supports decoding.",
      "Verify the exact tokenizer file, vocabulary size, special-token IDs, encode-decode behavior, and checkpoint compatibility.",
      "database_index", "A tokenizer maps text pieces to IDs; a database index maps search keys to record locations.",
      ("tokenizer", "token ids", "vocabulary", "decode")),
    C("token", "transformers_llms", "token", "model text unit",
      "A token is one discrete text unit represented by an integer ID for a language model.",
      "vocabulary", "A token is one unit in an encoded sequence; the vocabulary is the complete set of available token units.",
      "A punctuation mark, whole word, or word fragment can each be a token.",
      "Tokens are the units over which context length, embeddings, prediction, and generation are defined.",
      "The tokenizer maps text to token IDs, and the model looks up an embedding for each ID.",
      "Inspect the token IDs, decoded pieces, special tokens, unexpected fragmentation, and sequence truncation.",
      "database_row", "A database row stores a record; a token is one encoded unit in a model sequence.",
      ("token", "token id", "text unit", "sequence")),
    C("vocabulary", "transformers_llms", "model vocabulary", "set of available tokens",
      "A model vocabulary is the fixed mapping between supported token strings and integer token IDs.",
      "token", "A token is one encoded unit; the vocabulary defines every token unit the tokenizer and model can use.",
      "A vocabulary may contain common words, subwords, punctuation, bytes, and special control tokens.",
      "Vocabulary design affects text coverage, sequence length, embedding size, and checkpoint compatibility.",
      "The tokenizer selects entries from the vocabulary and the model uses their IDs to index embedding rows.",
      "Check vocabulary size, duplicate IDs, missing special tokens, tokenizer-checkpoint agreement, and out-of-range IDs.",
      "database_schema", "A database schema defines stored structure; a model vocabulary defines available token IDs.",
      ("vocabulary", "token ids", "mapping", "special tokens")),
    C("embedding", "transformers_llms", "embedding", "learned vector representation",
      "An embedding is a learned numeric vector used to represent a token or other discrete item.",
      "positional_encoding", "Token embeddings represent token identity; positional encoding adds information about sequence position.",
      "The token cat can be mapped to a dense vector before entering transformer blocks.",
      "Embeddings give discrete symbols a continuous representation that neural layers can transform.",
      "A token ID indexes a learned matrix row, producing a vector that is updated during training.",
      "Check ID ranges, embedding dimensions, weight tying, checkpoint shapes, and whether embeddings receive gradients.",
      "database_row", "A database row stores field values; an embedding is a learned dense vector.",
      ("embedding", "vector", "token id", "representation")),
    C("positional_encoding", "transformers_llms", "positional encoding", "sequence position signal",
      "Positional encoding adds information about token order to representations processed by a transformer.",
      "embedding", "Embeddings identify tokens; positional encodings distinguish where those tokens occur in the sequence.",
      "The same word at positions two and ten receives different combined position information.",
      "Attention alone is permutation-insensitive, so position signals are needed to represent order.",
      "Learned or computed position vectors are added to or combined with token representations.",
      "Check maximum sequence length, position offsets, checkpoint shape, padding behavior, and whether positions reset correctly.",
      "database_index", "A database index orders lookup keys; positional encoding marks sequence locations for a model.",
      ("positional encoding", "position", "token order", "sequence")),
    C("self_attention", "transformers_llms", "self-attention", "within-sequence attention",
      "Self-attention lets each token combine information from other tokens in the same sequence.",
      "multi_head_attention", "Self-attention describes the interaction; multi-head attention runs several learned attention projections in parallel.",
      "In The animal did not cross because it was tired, attention can connect it with animal.",
      "Self-attention allows context-dependent representations and direct interaction across distant positions.",
      "Queries and keys produce attention weights, which combine value vectors into a new representation for each token.",
      "Inspect query-key-value shapes, scaling, masks, softmax dimension, NaNs, and whether attention rows sum to one.",
      "database_join", "A database join combines records by matching keys; self-attention mixes token information by learned relevance.",
      ("self-attention", "query", "key", "value")),
    C("attention_head", "transformers_llms", "attention head", "single attention projection",
      "An attention head is one set of query, key, and value projections that computes an attention pattern.",
      "multi_head_attention", "An attention head is one channel; multi-head attention combines several heads.",
      "One head may focus strongly on nearby punctuation while another captures subject references.",
      "Separate heads give the layer multiple representation subspaces for contextual interactions.",
      "The head projects inputs to queries, keys, and values, applies scaled attention, and returns a weighted value mixture.",
      "Check per-head dimensions, reshaping and transpose order, mask broadcasting, output concatenation, and dead heads.",
      "database_column", "A database column stores one field; an attention head computes one learned interaction channel.",
      ("attention head", "query", "key", "value")),
    C("multi_head_attention", "transformers_llms", "multi-head attention", "parallel attention heads",
      "Multi-head attention runs multiple attention heads in parallel and combines their outputs.",
      "attention_head", "An attention head is one projection set; multi-head attention is the complete parallel module.",
      "A layer with eight heads can model several kinds of token relationships at once.",
      "Multiple heads let a model represent different interaction patterns in the same layer.",
      "Inputs are projected per head, attention is computed independently, outputs are concatenated, and a final projection mixes them.",
      "Verify head divisibility, reshape order, masks, concatenation dimensions, output projection, and gradient flow.",
      "database_view", "A database view exposes a saved query result; multi-head attention combines parallel attention computations.",
      ("multi-head attention", "heads", "parallel", "projection")),
    C("query_key_value", "transformers_llms", "query-key-value projections", "QKV projections",
      "Query, key, and value projections transform token representations into the vectors used by attention.",
      "self_attention", "QKV projections are the inputs to the attention calculation; self-attention includes weighting and value aggregation.",
      "A token's query is compared with every allowed key, then the resulting weights mix the values.",
      "QKV projections let attention learn what to seek, what to match, and what information to retrieve.",
      "Learned matrices create Q, K, and V; scaled QK dot products pass through softmax and weight V.",
      "Check matrix shapes, scaling by head dimension, transpose order, mask application, and softmax axis.",
      "sql_query", "A SQL query requests stored data; an attention query is a learned vector matched against keys.",
      ("query", "key", "value", "qkv")),
    C("causal_mask", "transformers_llms", "causal mask", "future-token mask",
      "A causal mask prevents a language model position from attending to future tokens.",
      "context_window", "A causal mask controls visibility direction; the context window controls how many tokens fit.",
      "While predicting token five, positions six and later are masked from attention.",
      "It prevents training leakage and preserves autoregressive next-token generation.",
      "Disallowed future attention logits receive a very negative value before softmax.",
      "Visualize the mask, verify diagonal orientation and broadcasting, and test that changing a future token cannot affect an earlier output.",
      "database_constraint", "A database constraint enforces data rules; a causal mask restricts attention to available past context.",
      ("causal mask", "future tokens", "autoregressive", "attention")),
    C("context_window", "transformers_llms", "context window", "maximum model context",
      "A context window is the maximum number of tokens a model can process together for one prediction.",
      "causal_mask", "The context window limits sequence length; a causal mask limits which positions within it are visible.",
      "A model with a 512-token context can use at most 512 recent tokens for a generation step.",
      "It determines how much prior text the model can directly use and affects memory and compute cost.",
      "Inputs longer than the supported block are truncated, chunked, or otherwise managed before the forward pass.",
      "Count tokens after formatting, inspect truncation side, verify block size and position limits, and test long prompts.",
      "database_table", "A database table stores many records; a context window is a bounded token sequence used by a model.",
      ("context window", "tokens", "block size", "sequence length")),
    C("feed_forward_network", "transformers_llms", "transformer feed-forward network", "MLP sublayer",
      "A transformer feed-forward network is a position-wise neural sublayer that expands, activates, and projects each token representation.",
      "self_attention", "Attention mixes information across positions; the feed-forward network transforms each position independently.",
      "A token vector may be expanded to four times the model width, passed through an activation, and projected back.",
      "It adds nonlinear feature transformation and much of a transformer's parameter capacity.",
      "Two learned linear projections surround a nonlinear activation and are applied to every sequence position.",
      "Check input-output dimensions, activation choice, dropout mode, residual addition, and gradient or activation scale.",
      "database_view", "A database view presents a query result; a feed-forward sublayer transforms token vectors.",
      ("feed-forward", "mlp", "activation", "projection")),
    C("residual_connection", "transformers_llms", "residual connection", "skip connection",
      "A residual connection adds a sublayer's input to its output.",
      "layer_normalization", "Residual connections preserve an information path; layer normalization rescales activations.",
      "A transformer block adds the attention output back to the representation that entered attention.",
      "Residual paths improve gradient flow and let deep networks learn incremental changes.",
      "The sublayer computes a transformation and its result is added elementwise to the original representation.",
      "Check matching tensor shapes, addition order, accidental in-place operations, dropout placement, and gradient flow.",
      "database_transaction", "A transaction groups database operations; a residual connection adds a neural sublayer update.",
      ("residual connection", "skip connection", "add", "gradient flow")),
    C("layer_normalization", "transformers_llms", "layer normalization", "activation normalization",
      "Layer normalization rescales each token's hidden features using statistics computed across its feature dimension.",
      "residual_connection", "Layer normalization stabilizes activation scale; residual connections preserve and add information paths.",
      "A transformer can normalize a token's hidden vector before applying attention.",
      "It improves optimization stability across deep transformer blocks.",
      "For each normalized vector, the layer subtracts its mean, divides by standard deviation, then applies learned scale and bias.",
      "Check the normalized dimension, epsilon, pre-norm versus post-norm order, checkpoint parameters, and NaNs.",
      "database_normalization", "Database normalization restructures tables; layer normalization rescales neural activations.",
      ("layer normalization", "mean", "variance", "activation")),
    C("softmax", "transformers_llms", "softmax", "probability normalization",
      "Softmax converts a vector of scores into positive values that sum to one.",
      "self_attention", "Softmax is a normalization operation used inside attention; it is not the whole attention mechanism.",
      "Next-token logits can pass through softmax to form a probability distribution over the vocabulary.",
      "It turns relative scores into normalized weights for selection or weighted averaging.",
      "Each exponentiated score is divided by the sum of exponentiated scores, usually after subtracting the maximum for stability.",
      "Check the dimension, temperature, masking before softmax, overflow handling, and whether outputs sum to one.",
      "sql", "SQL is a database language; softmax numerically normalizes model scores.",
      ("softmax", "probability", "scores", "sum to one")),
    C("next_token_prediction", "transformers_llms", "next-token prediction", "predicting the following token",
      "Next-token prediction trains or uses a language model to estimate which token should follow a context.",
      "pretraining", "Next-token prediction is an objective; pretraining is the broad training stage that may use that objective.",
      "Given The sky is, a model may assign high probability to blue.",
      "Predicting successive tokens teaches broad statistical patterns of language and supports generation.",
      "The model produces vocabulary logits at each position and compares them with the following token during training.",
      "Check target shifting, causal masking, padding loss masks, vocabulary alignment, and decoded top predictions.",
      "query_plan", "A query plan chooses database operations; next-token prediction scores possible following tokens.",
      ("next-token prediction", "following token", "logits", "language model")),
    C("pretraining", "transformers_llms", "language-model pretraining", "base model training",
      "Pretraining is the large-scale initial training stage that teaches a model general patterns from broad data.",
      "fine_tuning", "Pretraining builds broad capability; fine-tuning adapts the pretrained model to a narrower behavior or task.",
      "Training a transformer on a large text corpus with next-token prediction is pretraining.",
      "It provides reusable language and world-pattern representations before task-specific adaptation.",
      "The model processes many batches under a general objective and updates all or most parameters over many steps.",
      "Inspect corpus quality, tokenizer compatibility, loss curves, checkpoint integrity, learning rate, and sample generations.",
      "database", "A database stores application state; pretraining learns general model parameters from data.",
      ("pretraining", "base model", "corpus", "next-token")),
    C("fine_tuning", "transformers_llms", "fine-tuning", "adapting a pretrained model",
      "Fine-tuning continues training a pretrained model on focused data to change its behavior for a target task.",
      "pretraining", "Fine-tuning adapts an existing model with focused data; pretraining builds the broad base model.",
      "Supervised fine-tuning can train a base model to answer questions in a chat format.",
      "It can specialize behavior with far less data and compute than training a model from scratch.",
      "A compatible checkpoint is loaded and optimized on target examples with a small learning rate.",
      "Verify the base checkpoint, tokenizer, target masking, data format, learning rate, overfitting, and output checkpoint path.",
      "database_schema", "A database schema defines stored structure; fine-tuning changes learned model behavior.",
      ("fine-tuning", "pretrained", "sft", "adapt")),
    C("prompt", "transformers_llms", "prompt", "model input instruction",
      "A prompt is the input text or structured context provided to a language model to guide its output.",
      "context_window", "A prompt is the supplied content; the context window is the maximum token capacity containing that content.",
      "Summarize this paragraph in two sentences is a prompt.",
      "Prompt wording and context strongly influence which learned behavior the model activates.",
      "The prompt is tokenized, processed as context, and followed by generated or scored tokens.",
      "Inspect exact formatting, special tags, token count, ambiguous instructions, missing context, and decoding parameters.",
      "sql_query", "A SQL query formally requests database operations; a prompt supplies context and instructions to a language model.",
      ("prompt", "instruction", "input text", "context")),
    C("hallucination", "transformers_llms", "language-model hallucination", "unsupported generated claim",
      "A language-model hallucination is generated content that sounds plausible but is unsupported, false, or inconsistent with the provided context.",
      "next_token_prediction", "Next-token prediction generates likely continuations; hallucination is a failure where that continuation is unsupported.",
      "A model inventing a paper title and citation that do not exist is a hallucination.",
      "Hallucinations reduce reliability and can cause harmful decisions when users trust fluent output.",
      "The model selects statistically likely tokens without a built-in guarantee that every claim is grounded in verified evidence.",
      "Require sources, compare with trusted references, lower unsupported creativity, improve retrieval or data, and test known failure cases.",
      "database_constraint", "A database constraint enforces stored-data rules; hallucination is unsupported model-generated content.",
      ("hallucination", "unsupported", "false claim", "generated")),

    # Databases: 22 concepts.
    C("database", "databases", "database", "organized data store",
      "A database is an organized system for storing, retrieving, and managing data.",
      "relational_database", "A database is the general category; a relational database is one type organized around tables and relations.",
      "An application can store customer accounts and orders in a database.",
      "Databases provide durable, shared, and structured access to application data.",
      "A database engine writes data to managed storage and executes operations for reading, updating, and protecting it.",
      "Check connectivity, credentials, storage capacity, logs, locks, query behavior, and recent schema or configuration changes.",
      "artificial_intelligence", "A database stores and retrieves records; AI performs reasoning, prediction, or decision tasks.",
      ("database", "store data", "retrieve", "records")),
    C("relational_database", "databases", "relational database", "table-based database",
      "A relational database organizes data into tables connected by keys and queried with relational operations.",
      "database", "A relational database is a specific table-based database model within the broader database category.",
      "A shop can store customers, products, and orders in related SQL tables.",
      "Relational databases provide structured schemas, joins, constraints, and transaction guarantees.",
      "Rows are stored in tables, keys express relationships, and a query engine executes relational operations.",
      "Check schema definitions, key relationships, constraints, query plans, locks, and transaction behavior.",
      "transformer", "A relational database stores linked records; a transformer computes representations for sequences.",
      ("relational database", "tables", "keys", "sql")),
    C("database_table", "databases", "database table", "table",
      "A database table is a named collection of rows that share a defined set of columns.",
      "database_row", "A table is the whole collection and structure; a row is one record inside it.",
      "A users table may contain id, email, and created_at columns.",
      "Tables group related records under a consistent schema for querying and integrity checks.",
      "The database stores each record as a row whose values correspond to the table's columns.",
      "Inspect the schema, column types, constraints, permissions, row counts, indexes, and recent migrations.",
      "training_data", "A table stores application records; training data is a selected set of examples used to fit a model.",
      ("database table", "table", "rows", "columns")),
    C("database_row", "databases", "database row", "record",
      "A database row is one record containing values for the columns of a table.",
      "database_column", "A row contains one entity or event record; a column defines one field across many rows.",
      "One users row may contain id 42, an email address, and a creation timestamp.",
      "Rows are the individual records applications insert, retrieve, update, and delete.",
      "Each row stores column-aligned values and may be identified or linked through keys.",
      "Check the primary key, column values and types, constraints, transaction visibility, and whether filters select the intended row.",
      "token", "A token is one model text unit; a database row is one stored record.",
      ("database row", "row", "record", "values")),
    C("database_column", "databases", "database column", "table field",
      "A database column defines one named field and data type shared by rows in a table.",
      "database_row", "A column describes one field across records; a row contains the field values for one record.",
      "An email column can store text values for every row in a users table.",
      "Columns give stored values names, types, and optional integrity rules.",
      "The schema defines the column and each row stores a compatible value or null according to its constraints.",
      "Check the declared type, nullability, default, constraints, application mapping, and migration history.",
      "feature", "A feature is a model input variable; a database column is a persistent storage field.",
      ("database column", "column", "field", "data type")),
    C("primary_key", "databases", "primary key", "unique row identifier",
      "A primary key is a column or column set that uniquely identifies each row in a table.",
      "foreign_key", "A primary key identifies rows in its own table; a foreign key references a key in another or the same table.",
      "A users table can use user_id as its primary key.",
      "Primary keys prevent duplicate identity and provide stable targets for references and updates.",
      "The database enforces uniqueness and non-null values for the selected key columns.",
      "Check duplicate or null candidates, sequence generation, type mismatches, composite-key order, and conflicting inserts.",
      "label", "A label is a supervised-learning target; a primary key identifies a stored row.",
      ("primary key", "unique", "identify", "row")),
    C("foreign_key", "databases", "foreign key", "referential key",
      "A foreign key is a column or column set whose values reference a key in another or the same table.",
      "primary_key", "A foreign key creates a reference; a primary key supplies a unique row identity.",
      "An orders.customer_id foreign key can reference customers.customer_id.",
      "Foreign keys preserve valid relationships and prevent references to missing records.",
      "The database checks inserted or updated values against the referenced key and applies configured delete or update actions.",
      "Check referenced values, matching data types, insert order, cascade rules, deferred constraints, and orphaned legacy data.",
      "recall", "Recall measures found positive cases; a foreign key links database records.",
      ("foreign key", "reference", "referential", "key")),
    C("sql", "databases", "SQL", "structured query language",
      "SQL is a language for defining, querying, and modifying data in relational database systems.",
      "sql_query", "SQL is the language; a SQL query is one statement written in that language.",
      "SELECT email FROM users WHERE active = true is SQL.",
      "SQL provides a declarative way to work with structured data and database schemas.",
      "A parser and optimizer translate a SQL statement into operations the database engine executes.",
      "Check syntax, schema names, permissions, parameter binding, data types, and the generated query plan.",
      "softmax", "SQL expresses database operations; softmax converts numeric model scores into normalized weights.",
      ("sql", "select", "relational", "statement")),
    C("sql_query", "databases", "SQL query", "database query",
      "A SQL query is a statement that asks a relational database to read or manipulate data.",
      "sql", "SQL is the complete language; a query is one SQL statement or request.",
      "SELECT * FROM orders WHERE total > 100 is a SQL query.",
      "Queries are the primary interface for retrieving and changing relational data.",
      "The database parses the statement, optimizes an execution plan, runs operators, and returns a result or change count.",
      "Check parameters, filters, joins, permissions, indexes, query plan, lock waits, and returned row counts.",
      "inference", "Model inference computes a learned prediction; a SQL query operates on stored records.",
      ("sql query", "query", "select", "result")),
    C("database_index", "databases", "database index", "index",
      "A database index is an auxiliary data structure that speeds row lookup by organizing selected column values.",
      "b_tree_index", "A database index is the general feature; a B-tree index is one common ordered index structure.",
      "An index on users.email can speed searches for a specific email address.",
      "Indexes reduce the amount of table data a query must scan, trading extra storage and write work for faster reads.",
      "The index maps ordered or hashed key values to row locations so matching records can be found efficiently.",
      "Inspect the query plan, indexed column order, selectivity, stale statistics, missing predicates, size, and write overhead.",
      "tokenization", "A database index organizes lookup keys for rows; tokenization splits text into model units.",
      ("database index", "index", "lookup", "query")),
    C("b_tree_index", "databases", "B-tree index", "balanced tree index",
      "A B-tree index is a balanced ordered tree structure used for database lookups and range scans.",
      "database_index", "A database index is the broad category; a B-tree is a specific index implementation.",
      "A B-tree on created_at can efficiently find rows from a date range.",
      "Its ordered balanced structure supports equality, prefix, ordering, and range operations with predictable depth.",
      "Internal nodes guide searches by key ranges and leaf entries point to rows or contain indexed values.",
      "Check column order, predicate shape, data distribution, page splits, index bloat, statistics, and the query plan.",
      "gradient_descent", "Gradient descent updates model parameters; a B-tree index organizes database keys.",
      ("b-tree", "index", "range scan", "balanced tree")),
    C("full_table_scan", "databases", "full table scan", "sequential table scan",
      "A full table scan reads all or most rows of a table to evaluate a query.",
      "database_index", "A full scan examines the table broadly; an index lookup narrows access through indexed keys.",
      "A query without a useful index may scan every orders row to find one customer.",
      "Scans are sometimes optimal for small tables or broad results but can be expensive for selective queries on large tables.",
      "The engine reads table pages and tests each visible row against the query predicates.",
      "Inspect the query plan, estimated versus actual rows, filter selectivity, available indexes, statistics, and table size.",
      "underfitting", "Underfitting is inadequate model learning; a full table scan is a database access strategy.",
      ("full table scan", "sequential scan", "all rows", "query plan")),
    C("database_join", "databases", "database join", "joining tables",
      "A database join combines rows from two inputs according to a matching condition.",
      "foreign_key", "A foreign key defines and enforces a relationship; a join retrieves combined rows using a condition.",
      "Joining orders to customers on customer_id can return each order with the customer's email.",
      "Joins let normalized data remain separate while queries reconstruct useful combined views.",
      "The engine chooses a nested-loop, hash, merge, or other join algorithm based on data and indexes.",
      "Check join keys and types, duplicate matches, null behavior, filters, indexes, cardinality estimates, and the query plan.",
      "self_attention", "Self-attention mixes token representations by learned weights; a database join matches stored rows by a condition.",
      ("database join", "join", "matching", "rows")),
    C("database_transaction", "databases", "database transaction", "transaction",
      "A database transaction is a group of operations treated as one logical unit of work.",
      "acid", "A transaction is the unit of work; ACID names properties that make transaction behavior reliable.",
      "Transferring money can debit one account and credit another in a single transaction.",
      "Transactions protect multi-step updates from partial completion and unsafe concurrency.",
      "The database begins a transaction, applies statements under isolation, and then commits all changes or rolls them back.",
      "Check open transactions, commit and rollback paths, isolation level, lock waits, retries, timeouts, and error handling.",
      "reinforcement_learning", "Reinforcement learning optimizes sequential actions from rewards; a transaction groups database operations.",
      ("transaction", "commit", "rollback", "unit of work")),
    C("acid", "databases", "ACID properties", "transaction reliability properties",
      "ACID stands for atomicity, consistency, isolation, and durability, four properties associated with reliable database transactions.",
      "database_transaction", "A transaction is an operation group; ACID describes the guarantees expected around that group.",
      "Atomicity ensures a two-step funds transfer does not commit only one side.",
      "ACID properties reduce corruption and surprising behavior during failures and concurrent updates.",
      "The database uses logging, locking or versioning, constraints, and recovery mechanisms to provide the guarantees.",
      "Identify which property appears violated, then inspect isolation settings, transaction boundaries, logs, constraints, and crash recovery.",
      "model_evaluation", "Model evaluation measures predictive quality; ACID describes transaction guarantees.",
      ("acid", "atomicity", "isolation", "durability")),
    C("database_normalization", "databases", "database normalization", "normalizing a schema",
      "Database normalization organizes relational tables to reduce redundancy and update anomalies.",
      "denormalization", "Normalization separates repeated facts for integrity; denormalization deliberately duplicates or combines data for access speed.",
      "Moving customer details out of every order row into a customers table is normalization.",
      "It improves consistency by storing each fact in a controlled place.",
      "Designers identify dependencies and decompose tables into related structures that satisfy chosen normal forms.",
      "Check functional dependencies, duplicate facts, update anomalies, missing keys, excessive joins, and workload requirements.",
      "layer_normalization", "Layer normalization rescales neural activations; database normalization restructures relational tables.",
      ("database normalization", "normal form", "redundancy", "schema")),
    C("denormalization", "databases", "denormalization", "intentional schema duplication",
      "Denormalization intentionally duplicates or combines data to make some reads simpler or faster.",
      "database_normalization", "Denormalization trades redundancy for read convenience; normalization reduces redundancy for integrity.",
      "Storing a customer's current name on an analytics fact row can avoid a frequent join.",
      "It can improve read-heavy workloads when joins or recomputation are too expensive.",
      "Selected derived or repeated values are stored and kept synchronized through application or database processes.",
      "Check stale copies, update paths, storage growth, write amplification, consistency rules, and whether query gains are measured.",
      "fine_tuning", "Fine-tuning adapts model weights; denormalization changes how database facts are stored.",
      ("denormalization", "duplicate data", "read performance", "redundancy")),
    C("database_schema", "databases", "database schema", "stored data structure",
      "A database schema defines tables, columns, relationships, constraints, and other structural objects.",
      "database_table", "A table is one schema object; the schema describes the broader database structure.",
      "A schema can define users and orders tables linked by a customer foreign key.",
      "The schema gives applications and the database a shared contract for data shape and integrity.",
      "Definition statements create named objects, types, keys, constraints, indexes, and relationships.",
      "Compare the expected and actual migration version, object names, types, constraints, permissions, and application mappings.",
      "vocabulary", "A model vocabulary maps token IDs; a database schema defines persistent data structures.",
      ("database schema", "tables", "columns", "structure")),
    C("database_constraint", "databases", "database constraint", "data integrity rule",
      "A database constraint is a rule enforced by the database to restrict stored values or relationships.",
      "database_schema", "The schema is the whole structural definition; constraints are specific integrity rules within it.",
      "A unique constraint can prevent two accounts from using the same email address.",
      "Constraints keep invalid data out even when several applications write to the database.",
      "The database checks the declared rule during relevant inserts, updates, or transaction completion.",
      "Read the exact violation, identify the constraint columns, inspect conflicting data, null behavior, and transaction order.",
      "causal_mask", "A causal mask restricts model attention; a database constraint restricts stored data.",
      ("database constraint", "constraint", "integrity", "rule")),
    C("database_view", "databases", "database view", "saved query interface",
      "A database view is a named query that presents data like a virtual table.",
      "database_table", "A table stores rows directly; a standard view derives its rows from a query.",
      "An active_users view can expose only users whose status is active.",
      "Views simplify repeated queries, provide stable interfaces, and can restrict exposed columns or rows.",
      "When queried, the database expands or executes the view definition against its underlying objects.",
      "Check the view definition, underlying object changes, permissions, filters, column names, and the expanded query plan.",
      "multi_head_attention", "Multi-head attention combines neural attention heads; a database view exposes a named query result.",
      ("database view", "view", "virtual table", "query")),
    C("query_plan", "databases", "query plan", "database execution plan",
      "A query plan is the sequence of operations a database chooses to execute a query.",
      "sql_query", "A SQL query states the requested result; the query plan describes how the engine will produce it.",
      "A plan may use an index scan, then a nested-loop join, then a sort.",
      "The plan reveals where database work occurs and is central to diagnosing slow queries.",
      "The optimizer estimates alternative access and join strategies and selects a plan based on cost and statistics.",
      "Compare estimated and actual rows, scan and join choices, sort or spill costs, indexes, statistics, and lock time.",
      "next_token_prediction", "Next-token prediction scores possible tokens; a query plan chooses database execution operators.",
      ("query plan", "execution plan", "optimizer", "scan")),
    C("deadlock", "databases", "database deadlock", "cyclic lock wait",
      "A database deadlock occurs when transactions wait on one another in a cycle and none can proceed.",
      "database_transaction", "A transaction is a unit of work; a deadlock is a concurrency failure involving multiple transactions.",
      "Transaction A holds row one and waits for row two while transaction B holds row two and waits for row one.",
      "Deadlocks can abort work and reduce reliability under concurrent load.",
      "Conflicting lock acquisition orders create a wait cycle until the database detects it and cancels a participant.",
      "Inspect the deadlock graph, lock order, transaction duration, touched rows, indexes, retry policy, and consistent access ordering.",
      "gradient_descent", "Gradient descent is model optimization; a deadlock is a cyclic wait among database transactions.",
      ("deadlock", "lock", "transactions", "wait cycle")),
)


TRAIN_QUESTION_TEMPLATES = {
    "definition": {
        "normal": "What is {label}?",
        "paraphrase": "Explain {alias} in simple terms.",
        "contrast": "How is {label} different from {contrast_label}?",
    },
    "comparison": {
        "normal": "Compare {label} with {related_label}.",
        "paraphrase": "What distinguishes {alias} from {related_label}?",
        "contrast": "Why should {label} not be confused with {contrast_label}?",
    },
    "example": {
        "normal": "Give a concrete example of {label}.",
        "paraphrase": "Show how {alias} appears in a practical scenario.",
        "contrast": "Give an example of {label} that would not be an example of {contrast_label}.",
    },
    "why": {
        "normal": "Why is {label} important?",
        "paraphrase": "What problem makes {alias} useful?",
        "contrast": "Why would someone use or study {label} instead of {contrast_label}?",
    },
    "how": {
        "normal": "How does {label} work?",
        "paraphrase": "Describe the main process behind {alias}.",
        "contrast": "How does {label} work differently from {contrast_label}?",
    },
    "troubleshooting": {
        "normal": "How should you troubleshoot {label}?",
        "paraphrase": "What should you check when {alias} is not working as expected?",
        "contrast": "If a system confuses {label} with {contrast_label}, how do you diagnose the mistake?",
    },
}

HELD_OUT_QUESTION_TEMPLATES = {
    "definition": {
        "normal": "State a precise one-sentence definition of {label}.",
        "paraphrase": "For a new learner, what does {alias} mean?",
        "contrast": "A teammate treats {label} and {contrast_label} as the same thing. Correct that misunderstanding.",
    },
    "comparison": {
        "normal": "What is the key boundary between {label} and {related_label}?",
        "paraphrase": "When choosing terminology, how can I tell {alias} apart from {related_label}?",
        "contrast": "Contrast {label} with {contrast_label} without mixing their roles.",
    },
    "example": {
        "normal": "Describe one realistic situation that clearly demonstrates {label}.",
        "paraphrase": "What practical case would help someone recognize {alias}?",
        "contrast": "Provide a case that is specifically {label}, not {contrast_label}.",
    },
    "why": {
        "normal": "What limitation or need does {label} address?",
        "paraphrase": "Why does {alias} matter in a real system?",
        "contrast": "What goal calls for {label} rather than {contrast_label}?",
    },
    "how": {
        "normal": "Walk through the core mechanism of {label}.",
        "paraphrase": "At a high level, what steps make {alias} function?",
        "contrast": "Explain the operating difference between {label} and {contrast_label}.",
    },
    "troubleshooting": {
        "normal": "{label} is producing unexpected results. Which diagnostics should come first?",
        "paraphrase": "How would you investigate a suspected failure in {alias}?",
        "contrast": "An answer routes {label} to {contrast_label}. What evidence would expose and fix that routing error?",
    },
}


def concept_map() -> dict[str, Concept]:
    concepts = {concept.key: concept for concept in CONCEPTS}
    if len(concepts) != len(CONCEPTS):
        raise ValueError("Concept keys must be unique")
    return concepts


def display_label(key: str, concepts: dict[str, Concept]) -> str:
    if key in concepts:
        return concepts[key].label
    return key.replace("_", " ")


def format_question(
    concept: Concept,
    question_type: str,
    routing_variant: str,
    templates: dict[str, dict[str, str]],
    concepts: dict[str, Concept],
) -> str:
    return templates[question_type][routing_variant].format(
        label=concept.label,
        alias=concept.alias,
        related_label=display_label(concept.related, concepts),
        contrast_label=display_label(concept.contrast, concepts),
    )


def base_response(concept: Concept, question_type: str) -> str:
    return {
        "definition": concept.definition,
        "comparison": f"{concept.definition} {concept.comparison}",
        "example": f"{concept.definition} {concept.example}",
        "why": f"{concept.definition} {concept.importance}",
        "how": f"{concept.definition} {concept.mechanism}",
        "troubleshooting": f"{concept.definition} To troubleshoot it, {concept.troubleshooting[0].lower() + concept.troubleshooting[1:]}",
    }[question_type]


def response_for(concept: Concept, question_type: str, routing_variant: str) -> str:
    response = base_response(concept, question_type)
    if routing_variant == "contrast":
        response = f"{response} {concept.contrast_note}"
    return response


def hard_negatives_for(concept: Concept, concepts: dict[str, Concept]) -> list[str]:
    domain_concepts = [item.key for item in CONCEPTS if item.domain == concept.domain]
    index = domain_concepts.index(concept.key)
    candidates = [concept.related, concept.contrast, domain_concepts[(index + 1) % len(domain_concepts)]]
    result: list[str] = []
    for candidate in candidates:
        if candidate != concept.key and candidate not in result:
            result.append(candidate)
    for candidate in domain_concepts:
        if len(result) == 3:
            break
        if candidate != concept.key and candidate not in result:
            result.append(candidate)
    if len(result) != 3:
        raise ValueError(f"Could not assign three hard negatives for {concept.key}")
    return result


def build_sft() -> list[dict[str, object]]:
    concepts = concept_map()
    records: list[dict[str, object]] = []
    by_domain = defaultdict(list)
    for concept in CONCEPTS:
        by_domain[concept.domain].append(concept)
        negatives = hard_negatives_for(concept, concepts)
        for question_type in QUESTION_TYPES:
            for routing_variant in ROUTING_VARIANTS:
                records.append(
                    {
                        "instruction": format_question(
                            concept, question_type, routing_variant, TRAIN_QUESTION_TEMPLATES, concepts
                        ),
                        "response": response_for(concept, question_type, routing_variant),
                        "domain": concept.domain,
                        "concept": concept.key,
                        "question_type": question_type,
                        "routing_variant": routing_variant,
                        "related_concept": concept.related,
                        "contrast_concept": concept.contrast if routing_variant == "contrast" else None,
                        "hard_negative_concepts": negatives,
                        "keywords": list(concept.keywords),
                        "source": "marshmello_core_sft_v1",
                        "split": "train",
                    }
                )

    # 22 concepts x 6 question types x 3 routing variants = 396 per domain.
    # Add four unique phrasings per domain so each domain lands exactly at 400.
    extra_specs = (
        ("definition", "normal", "Give the shortest accurate explanation of {label}."),
        ("comparison", "paraphrase", "In one clear distinction, separate {alias} from {related_label}."),
        ("example", "contrast", "Illustrate {label} with a case that cannot be mistaken for {contrast_label}."),
        ("why", "normal", "What practical value does {label} provide?"),
    )
    for domain in DOMAINS:
        for concept, (question_type, routing_variant, template) in zip(by_domain[domain], extra_specs):
            negatives = hard_negatives_for(concept, concepts)
            instruction = template.format(
                label=concept.label,
                alias=concept.alias,
                related_label=display_label(concept.related, concepts),
                contrast_label=display_label(concept.contrast, concepts),
            )
            records.append(
                {
                    "instruction": instruction,
                    "response": response_for(concept, question_type, routing_variant),
                    "domain": concept.domain,
                    "concept": concept.key,
                    "question_type": question_type,
                    "routing_variant": routing_variant,
                    "related_concept": concept.related,
                    "contrast_concept": concept.contrast if routing_variant == "contrast" else None,
                    "hard_negative_concepts": negatives,
                    "keywords": list(concept.keywords),
                    "source": "marshmello_core_sft_v1",
                    "split": "train",
                }
            )
    # The existing trainer takes the final 5% as validation without shuffling.
    # Put exactly 20 records per domain at the end so that validation is balanced.
    validation_indices: set[int] = set()
    for domain in DOMAINS:
        domain_concepts = [concept for concept in CONCEPTS if concept.domain == domain]
        for i, concept in enumerate(domain_concepts[:20]):
            desired_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]
            desired_variant = ROUTING_VARIANTS[i % len(ROUTING_VARIANTS)]
            index = next(
                j
                for j, record in enumerate(records)
                if j not in validation_indices
                and record["domain"] == domain
                and record["concept"] == concept.key
                and record["question_type"] == desired_type
                and record["routing_variant"] == desired_variant
            )
            validation_indices.add(index)

    training_records = [
        {**record, "split": "train"}
        for i, record in enumerate(records)
        if i not in validation_indices
    ]
    validation_records = [
        {**record, "split": "validation"}
        for i, record in enumerate(records)
        if i in validation_indices
    ]
    return training_records + validation_records


def build_negatives() -> list[dict[str, object]]:
    concepts = concept_map()
    records: list[dict[str, object]] = []
    for concept in CONCEPTS:
        negatives = hard_negatives_for(concept, concepts)
        for question_type in QUESTION_TYPES:
            instruction = format_question(
                concept, question_type, "normal", TRAIN_QUESTION_TEMPLATES, concepts
            )
            records.append(
                {
                    "instruction": instruction,
                    "domain": concept.domain,
                    "concept": concept.key,
                    "question_type": question_type,
                    "correct_keywords": list(concept.keywords),
                    "incorrect_concepts": negatives,
                    "incorrect_domains": [
                        concepts[item].domain if item in concepts else "cross_domain" for item in negatives
                    ],
                    "source": "marshmello_core_hard_negatives_v1",
                }
            )
    return records


def build_eval() -> list[dict[str, object]]:
    concepts = concept_map()
    records: list[dict[str, object]] = []
    global_index = 0
    for domain in DOMAINS:
        domain_concepts = [concept for concept in CONCEPTS if concept.domain == domain]
        count = EVAL_DISTRIBUTION[domain]
        for i in range(count):
            concept = domain_concepts[i % len(domain_concepts)]
            question_type = QUESTION_TYPES[(global_index + i) % len(QUESTION_TYPES)]
            routing_variant = ROUTING_VARIANTS[(global_index + 2 * i) % len(ROUTING_VARIANTS)]
            negatives = hard_negatives_for(concept, concepts)
            records.append(
                {
                    "id": f"core-eval-{len(records) + 1:03d}",
                    "instruction": format_question(
                        concept, question_type, routing_variant, HELD_OUT_QUESTION_TEMPLATES, concepts
                    ),
                    "response": response_for(concept, question_type, routing_variant),
                    "domain": concept.domain,
                    "concept": concept.key,
                    "question_type": question_type,
                    "routing_variant": routing_variant,
                    "related_concept": concept.related,
                    "contrast_concept": concept.contrast if routing_variant == "contrast" else None,
                    "hard_negative_concepts": negatives,
                    "keywords": list(concept.keywords),
                    "source": "marshmello_core_eval_v1",
                    "split": "held_out",
                }
            )
        global_index += count
    return records


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate(
    sft: list[dict[str, object]],
    negatives: list[dict[str, object]],
    eval_records: list[dict[str, object]],
) -> None:
    concepts = concept_map()
    if {concept.domain for concept in CONCEPTS} != set(DOMAINS):
        raise ValueError("Unexpected concept domain")
    for domain in DOMAINS:
        count = sum(record["domain"] == domain for record in sft)
        if count != TARGET_PER_DOMAIN:
            raise ValueError(f"{domain}: expected {TARGET_PER_DOMAIN} SFT examples, got {count}")
        concept_count = sum(concept.domain == domain for concept in CONCEPTS)
        if concept_count != 22:
            raise ValueError(f"{domain}: expected 22 concepts, got {concept_count}")

    if len(sft) != TARGET_PER_DOMAIN * len(DOMAINS):
        raise ValueError(f"Expected 1200 SFT examples, got {len(sft)}")
    validation_tail = sft[-60:]
    if any(record["split"] != "validation" for record in validation_tail):
        raise ValueError("The final 5% must be marked as validation")
    if any(record["split"] != "train" for record in sft[:-60]):
        raise ValueError("Only the final 5% may be marked as validation")
    for domain in DOMAINS:
        validation_count = sum(record["domain"] == domain for record in validation_tail)
        if validation_count != 20:
            raise ValueError(f"{domain}: expected 20 validation-tail examples, got {validation_count}")
    if len(eval_records) != 100:
        raise ValueError(f"Expected 100 eval examples, got {len(eval_records)}")
    if len(negatives) != len(CONCEPTS) * len(QUESTION_TYPES):
        raise ValueError(f"Expected {len(CONCEPTS) * len(QUESTION_TYPES)} negatives, got {len(negatives)}")

    for domain, expected in EVAL_DISTRIBUTION.items():
        actual = sum(record["domain"] == domain for record in eval_records)
        if actual != expected:
            raise ValueError(f"{domain}: expected {expected} eval examples, got {actual}")

    train_questions = {normalized(str(record["instruction"])) for record in sft}
    eval_questions = {normalized(str(record["instruction"])) for record in eval_records}
    overlap = train_questions & eval_questions
    if overlap:
        raise ValueError(f"Eval questions overlap training: {sorted(overlap)[:3]}")
    if len(train_questions) != len(sft):
        raise ValueError("Training instructions must be unique")
    if len(eval_questions) != len(eval_records):
        raise ValueError("Evaluation instructions must be unique")

    coverage = defaultdict(set)
    for record in sft:
        coverage[str(record["concept"])].add(str(record["routing_variant"]))
        if record["question_type"] not in QUESTION_TYPES:
            raise ValueError(f"Bad question type: {record['question_type']}")
        if record["domain"] not in DOMAINS:
            raise ValueError(f"Bad domain: {record['domain']}")
        if len(record["hard_negative_concepts"]) != 3:
            raise ValueError(f"Expected three hard negatives: {record}")
        if record["concept"] in record["hard_negative_concepts"]:
            raise ValueError(f"Correct concept appears as a hard negative: {record}")
        if not str(record["response"]).strip():
            raise ValueError("Empty response")
    for concept in concepts:
        if coverage[concept] != set(ROUTING_VARIANTS):
            raise ValueError(f"{concept}: incomplete routing coverage {coverage[concept]}")

    # Balance is exact except for the unavoidable remainder of four per domain.
    for domain in DOMAINS:
        counts = Counter(
            str(record["question_type"]) for record in sft if record["domain"] == domain
        )
        if max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"{domain}: unbalanced question types {counts}")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def distribution_table(records: list[dict[str, object]], field: str) -> list[tuple[str, int]]:
    counts = Counter(str(record[field]) for record in records)
    return sorted(counts.items())


def write_report(
    path: Path,
    sft: list[dict[str, object]],
    negatives: list[dict[str, object]],
    eval_records: list[dict[str, object]],
) -> None:
    lines = [
        "# Marshmello Core SFT Dataset Report",
        "",
        "This report is generated by `18J_marshmello_core_sft/build_marshmello_core_data.py`.",
        "",
        "## Summary",
        "",
        f"- Training examples: {len(sft)}",
        f"- Hard-negative records: {len(negatives)}",
        f"- Held-out evaluation questions: {len(eval_records)}",
        f"- Concepts: {len(CONCEPTS)} (22 per domain)",
        "- Tokenizer, architecture, and model configuration changes: none",
        "",
        "## Training distribution",
        "",
        "| Domain | Examples |",
        "|---|---:|",
    ]
    for domain in DOMAINS:
        lines.append(f"| {domain} | {sum(r['domain'] == domain for r in sft)} |")
    lines.extend(["", "| Question type | Examples |", "|---|---:|"])
    for key, count in distribution_table(sft, "question_type"):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "| Routing variant | Examples |", "|---|---:|"])
    for key, count in distribution_table(sft, "routing_variant"):
        lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "Every concept has normal, paraphrased, and contrast coverage. Each concept also has "
            "definition, comparison, example, why, how, and troubleshooting questions.",
            "",
            "## Held-out evaluation distribution",
            "",
            "| Domain | Questions |",
            "|---|---:|",
        ]
    )
    for domain in DOMAINS:
        lines.append(f"| {domain} | {sum(r['domain'] == domain for r in eval_records)} |")
    lines.extend(
        [
            "",
            "All evaluation instructions are unique and have zero normalized-text overlap with "
            "the training instructions.",
            "",
            "## Hard negatives",
            "",
            "There are six hard-negative records per concept, one per question type. Every record "
            "contains exactly three incorrect concepts, including a related concept and an explicit "
            "cross-domain routing contrast.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python3 18J_marshmello_core_sft/build_marshmello_core_data.py",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-output", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--negatives-output", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    sft = build_sft()
    negatives = build_negatives()
    eval_records = build_eval()
    validate(sft, negatives, eval_records)

    if not args.validate_only:
        write_jsonl(args.sft_output, sft)
        write_jsonl(args.negatives_output, negatives)
        write_jsonl(args.eval_output, eval_records)
        write_report(args.report_output, sft, negatives, eval_records)

    print("Marshmello core data validation passed")
    print(f"SFT:       {len(sft)} examples ({TARGET_PER_DOMAIN} per domain)")
    print(f"Negatives: {len(negatives)} records")
    print(f"Eval:      {len(eval_records)} held-out questions")


if __name__ == "__main__":
    main()
