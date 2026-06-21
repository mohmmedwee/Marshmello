# Phase 18K — General Benchmark Comparison

- Generated: 2026-06-21T10:25:28.372746+00:00
- Eval: `general_eval.jsonl` (500 held-out, greedy decoding)

## Overall

| Checkpoint | Domain score | Keyword recall | Token overlap | Hallucination | Empty | Repetition |
|---|---:|---:|---:|---:|---:|---:|
| base | 11.7% | 3.0% | 9.9% | 84.2% | 0.4% | 67.6% |
| best_18j_routing | 21.8% | 2.2% | 10.5% | 68.2% | 0.2% | 15.8% |
| teacher_latest | 22.5% | 2.2% | 10.8% | 64.2% | 0.4% | 15.8% |

## By benchmark bucket

### base

| Bucket | Domain score | Hallucination |
|---|---:|---:|
| ai | 13.6% | 79.0% |
| databases | 14.7% | 79.0% |
| general_knowledge | 7.8% | 92.0% |
| programming | 11.9% | 83.0% |
| system_design | 10.3% | 88.0% |

### best_18j_routing

| Bucket | Domain score | Hallucination |
|---|---:|---:|
| ai | 23.7% | 64.0% |
| databases | 24.6% | 62.0% |
| general_knowledge | 20.1% | 72.0% |
| programming | 19.8% | 72.0% |
| system_design | 20.9% | 71.0% |

### teacher_latest

| Bucket | Domain score | Hallucination |
|---|---:|---:|
| ai | 23.1% | 63.0% |
| databases | 25.7% | 51.0% |
| general_knowledge | 22.3% | 64.0% |
| programming | 20.9% | 71.0% |
| system_design | 20.5% | 72.0% |

## Sample outputs (10 per checkpoint)

### base

#### 18k-eval-ai-001

**Q:** Explain how an encoder handles syntactic features.

**Generated:** An example of a type of a type of text is the word "artificial intelligence" is a text that uses a text to the context of a text and generate a text that uses a text to generate a prompt.

**Reference:** The encoder in a text classification model transforms input sentences into a fixed-length vector representation. For instance, a transformer's encoder captures syntactic and semantic features of the input, enabling accurate sentiment classification even with varying sentence structures.

#### 18k-eval-ai-051

**Q:** How is a number line different from a coordinate system?

**Generated:** A group of different types of different types of different types of types of types of types of types of weights. These types of types of types of types of types of types of types of weightning types of weightning types of weightning and weightning types of weightning behavior and weights.

**Reference:** A number line is a 1D visual representation of numbers, while a coordinate system extends this to 2D or more dimensions. Both use numbers to locate points, but coordinate systems handle multiple axes.

#### 18k-eval-databases-001

**Q:** How does a primary key function within a database table?

**Generated:** A database is a field that stores records as a set of instructions that guide a model toward the desired answer. It helps the system answer or act more usefully.

**Reference:** A primary key uniquely identifies each row by enforcing uniqueness and preventing duplicate entries, allowing efficient data retrieval and relationships with foreign keys.

#### 18k-eval-databases-051

**Q:** Define ACID properties in database systems.

**Generated:** CID is a type of database that stores records as a database structure. It helps keep data useful, reliable, or fast to access.

**Reference:** In database systems, ACID properties ensure reliable transaction processing through Atomicity (all-or-nothing execution), Consistency (valid state transitions), Isolation (concurrent transactions' independence), and Durability (persistent post-commit data). These principles guarantee robustness in relational and transactional databases.

#### 18k-eval-programming-001

**Q:** How does the content type header differ from the caching header?

**Generated:** The main difference between a web browser and a web browser is a web browser that is a web browser and a web browser to run a web server. The web browser is a web server is a server that provides a web server to run the web server.

**Reference:** The content type header specifies the data format (e.g., JSON, HTML), while the caching header determines how long a response is stored in the browser.

#### 18k-eval-programming-051

**Q:** How does hash map lookup function?

**Generated:** Function is a reusable block of code that can be changed. It helps Python code stay clear and practical.

**Reference:** A hash map lookup efficiently finds a value by computing the key's hash and accessing the corresponding stored value.

#### 18k-eval-system_design-001

**Q:** Define 'signal' in Linux.

**Generated:** The term 'unsupervised learning is finding structure in data without labeled data without labels. It is useful for building models that work beyond one example.

**Reference:** A signal in Linux is a software interrupt used to notify a process of an event. Common signals include SIGKILL and SIGTERM, which trigger specific actions like termination or suspension.

#### 18k-eval-system_design-051

**Q:** What is SaaS?

**Generated:** Sft is a text format for a query results. It helps keep data useful, reliable, or fast to access.

**Reference:** SaaS hosts software applications over the internet, allowing users to access and use them through web browsers without local installation.

#### 18k-eval-general_knowledge-001

**Q:** Predicted bounding boxes have low IoU with ground truth. What should you check first?

**Generated:** To make a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack of a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a p

**Reference:** Verify the IoU threshold in your evaluation script. Lowering it to 0.3 may resolve discrepancies.

#### 18k-eval-general_knowledge-051

**Q:** How does a climate zone differ from a physical map?

**Generated:** A good idea for a person should have a person who is important to take their own life. This should have a positive impact on their own life. It should also be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able

**Reference:** Climate zones classify broad temperature and precipitation patterns, while physical maps visually depict these conditions through color gradients and contour lines.

### best_18j_routing

#### 18k-eval-ai-001

**Q:** Explain how an encoder handles syntactic features.

**Generated:** A transformer is a neural network architecture that is part of how language models read prompts and produce answers.

**Reference:** The encoder in a text classification model transforms input sentences into a fixed-length vector representation. For instance, a transformer's encoder captures syntactic and semantic features of the input, enabling accurate sentiment classification even with varying sentence structures.

#### 18k-eval-ai-051

**Q:** How is a number line different from a coordinate system?

**Generated:** A database index is a named field that combines rows from related to a table.

**Reference:** A number line is a 1D visual representation of numbers, while a coordinate system extends this to 2D or more dimensions. Both use numbers to locate points, but coordinate systems handle multiple axes.

#### 18k-eval-databases-001

**Q:** How does a primary key function within a database table?

**Generated:** A database is a named field that speeds each row defines one named field inside a table.

**Reference:** A primary key uniquely identifies each row by enforcing uniqueness and preventing duplicate entries, allowing efficient data retrieval and relationships with foreign keys.

#### 18k-eval-databases-051

**Q:** Define ACID properties in database systems.

**Generated:** ACID is a rule for reliable transactions.

**Reference:** In database systems, ACID properties ensure reliable transaction processing through Atomicity (all-or-nothing execution), Consistency (valid state transitions), Isolation (concurrent transactions' independence), and Durability (persistent post-commit data). These principles guarantee robustness in relational and transactional databases.

#### 18k-eval-programming-001

**Q:** How does the content type header differ from the caching header?

**Generated:** The design is a vector that sounds confident but is not used for a language model.

**Reference:** The content type header specifies the data format (e.g., JSON, HTML), while the caching header determines how long a response is stored in the browser.

#### 18k-eval-programming-051

**Q:** How does hash map lookup function?

**Generated:** Fine-tuning is a sequence of actions before a constraint a rule that prevents a constraint a rule that prevents a constraint a language model to the same constraint a language model.

**Reference:** A hash map lookup efficiently finds a value by computing the key's hash and accessing the corresponding stored value.

#### 18k-eval-system_design-001

**Q:** Define 'signal' in Linux.

**Generated:** Gradient descent is a method that updates inputs in a model's ability to perform a neural network to process a neural network to process inputs in a model.

**Reference:** A signal in Linux is a software interrupt used to notify a process of an event. Common signals include SIGKILL and SIGTERM, which trigger specific actions like termination or suspension.

#### 18k-eval-system_design-051

**Q:** What is SaaS?

**Generated:** A SQL is a query that combines rows from related tables.

**Reference:** SaaS hosts software applications over the internet, allowing users to access and use them through web browsers without local installation.

#### 18k-eval-general_knowledge-001

**Q:** Predicted bounding boxes have low IoU with ground truth. What should you check first?

**Generated:** Underfitting choose actions from a branch of general patterns.

**Reference:** Verify the IoU threshold in your evaluation script. Lowering it to 0.3 may resolve discrepancies.

#### 18k-eval-general_knowledge-051

**Q:** How does a climate zone differ from a physical map?

**Generated:** A token is a token or numeric values.

**Reference:** Climate zones classify broad temperature and precipitation patterns, while physical maps visually depict these conditions through color gradients and contour lines.

### teacher_latest

#### 18k-eval-ai-001

**Q:** Explain how an encoder handles syntactic features.

**Generated:** A transformer is a neural network architecture that uses attention to process token sequences with attention and feed-forward layers joined by residual paths.

**Reference:** The encoder in a text classification model transforms input sentences into a fixed-length vector representation. For instance, a transformer's encoder captures syntactic and semantic features of the input, enabling accurate sentiment classification even with varying sentence structures.

#### 18k-eval-ai-051

**Q:** How is a number line different from a coordinate system?

**Generated:** A database index is a named field that combines rows from related to a table.

**Reference:** A number line is a 1D visual representation of numbers, while a coordinate system extends this to 2D or more dimensions. Both use numbers to locate points, but coordinate systems handle multiple axes.

#### 18k-eval-databases-001

**Q:** How does a primary key function within a database table?

**Generated:** A database is a named field that speeds each row defines one named field inside a table.

**Reference:** A primary key uniquely identifies each row by enforcing uniqueness and preventing duplicate entries, allowing efficient data retrieval and relationships with foreign keys.

#### 18k-eval-databases-051

**Q:** Define ACID properties in database systems.

**Generated:** ACID AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

**Reference:** In database systems, ACID properties ensure reliable transaction processing through Atomicity (all-or-nothing execution), Consistency (valid state transitions), Isolation (concurrent transactions' independence), and Durability (persistent post-commit data). These principles guarantee robustness in relational and transactional databases.

#### 18k-eval-programming-001

**Q:** How does the content type header differ from the caching header?

**Generated:** The primary key is a defined way that unsupervised learning trains paired with target answers.

**Reference:** The content type header specifies the data format (e.g., JSON, HTML), while the caching header determines how long a response is stored in the browser.

#### 18k-eval-programming-051

**Q:** How does hash map lookup function?

**Generated:** A SQL is a rule for relational database columns for relational database columns.

**Reference:** A hash map lookup efficiently finds a value by computing the key's hash and accessing the corresponding stored value.

#### 18k-eval-system_design-001

**Q:** Define 'signal' in Linux.

**Generated:** Softmax is a language used for a language model.

**Reference:** A signal in Linux is a software interrupt used to notify a process of an event. Common signals include SIGKILL and SIGTERM, which trigger specific actions like termination or suspension.

#### 18k-eval-system_design-051

**Q:** What is SaaS?

**Generated:** NER is the SQL query to read or used data.

**Reference:** SaaS hosts software applications over the internet, allowing users to access and use them through web browsers without local installation.

#### 18k-eval-general_knowledge-001

**Q:** Predicted bounding boxes have low IoU with ground truth. What should you check first?

**Generated:** Overfitting choose training examples used to teach a model to teach a classifier from past cases that are too little to follow a model to follow a prediction.

**Reference:** Verify the IoU threshold in your evaluation script. Lowering it to 0.3 may resolve discrepancies.

#### 18k-eval-general_knowledge-051

**Q:** How does a climate zone differ from a physical map?

**Generated:** A token is one full of a token or other numeric values.

**Reference:** Climate zones classify broad temperature and precipitation patterns, while physical maps visually depict these conditions through color gradients and contour lines.

## How to read this with 18J

- **18J** measures core concept routing (currently ~18% on `best_18j_routing`).
- **18K** measures general assistant quality on held-out broad SFT-style questions.
- A useful assistant can have low 18J routing but higher 18K scores after broad SFT.
- A 300M decision should consider **both** benchmarks, not 18J alone.

