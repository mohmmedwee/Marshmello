# Phase 18K — General Benchmark (best_18j_routing)

- Generated: 2026-06-21T10:23:49.314980+00:00
- Checkpoint: `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/18B_marshmello_instruct/checkpoints/best_18j_routing.pt`
- Questions: 500
- Decoding: greedy

## Overall metrics

| Metric | Value |
|---|---:|
| Keyword recall | 2.2% |
| Reference token overlap | 10.5% |
| Response length sanity | 99.8% |
| Repetition rate | 15.8% |
| Empty response rate | 0.2% |
| Hallucination proxy | 68.2% |
| Domain score (composite) | 21.8% |

## By benchmark bucket

| Bucket | Domain score | Keyword | Overlap | Hallucination |
|---|---:|---:|---:|---:|
| ai | 23.7% | 4.0% | 12.1% | 64.0% |
| databases | 24.6% | 5.1% | 12.8% | 62.0% |
| general_knowledge | 20.1% | 0.7% | 9.6% | 72.0% |
| programming | 19.8% | 0.4% | 8.7% | 72.0% |
| system_design | 20.9% | 0.7% | 9.2% | 71.0% |

## Sample outputs (10)

### 18k-eval-ai-001

**Q:** Explain how an encoder handles syntactic features.

**Reference:** The encoder in a text classification model transforms input sentences into a fixed-length vector representation. For instance, a transformer's encoder captures syntactic and semantic features of the input, enabling accurate sentiment classification even with varying sentence structures.

**Generated:** A transformer is a neural network architecture that is part of how language models read prompts and produce answers.

- keyword recall: 0.0%, overlap: 14.0%, domain score: 34.9%

### 18k-eval-ai-051

**Q:** How is a number line different from a coordinate system?

**Reference:** A number line is a 1D visual representation of numbers, while a coordinate system extends this to 2D or more dimensions. Both use numbers to locate points, but coordinate systems handle multiple axes.

**Generated:** A database index is a named field that combines rows from related to a table.

- keyword recall: 0.0%, overlap: 20.8%, domain score: 37.3%

### 18k-eval-databases-001

**Q:** How does a primary key function within a database table?

**Reference:** A primary key uniquely identifies each row by enforcing uniqueness and preventing duplicate entries, allowing efficient data retrieval and relationships with foreign keys.

**Generated:** A database is a named field that speeds each row defines one named field inside a table.

- keyword recall: 10.0%, overlap: 15.0%, domain score: 38.7%

### 18k-eval-databases-051

**Q:** Define ACID properties in database systems.

**Reference:** In database systems, ACID properties ensure reliable transaction processing through Atomicity (all-or-nothing execution), Consistency (valid state transitions), Isolation (concurrent transactions' independence), and Durability (persistent post-commit data). These principles guarantee robustness in relational and transactional databases.

**Generated:** ACID is a rule for reliable transactions.

- keyword recall: 20.0%, overlap: 13.3%, domain score: 41.7%

### 18k-eval-programming-001

**Q:** How does the content type header differ from the caching header?

**Reference:** The content type header specifies the data format (e.g., JSON, HTML), while the caching header determines how long a response is stored in the browser.

**Generated:** The design is a vector that sounds confident but is not used for a language model.

- keyword recall: 0.0%, overlap: 14.3%, domain score: 35.0%

### 18k-eval-programming-051

**Q:** How does hash map lookup function?

**Reference:** A hash map lookup efficiently finds a value by computing the key's hash and accessing the corresponding stored value.

**Generated:** Fine-tuning is a sequence of actions before a constraint a rule that prevents a constraint a rule that prevents a constraint a language model to the same constraint a language model.

- keyword recall: 0.0%, overlap: 11.8%, domain score: 4.1%

### 18k-eval-system_design-001

**Q:** Define 'signal' in Linux.

**Reference:** A signal in Linux is a software interrupt used to notify a process of an event. Common signals include SIGKILL and SIGTERM, which trigger specific actions like termination or suspension.

**Generated:** Gradient descent is a method that updates inputs in a model's ability to perform a neural network to process a neural network to process inputs in a model.

- keyword recall: 10.0%, overlap: 24.1%, domain score: 11.9%

### 18k-eval-system_design-051

**Q:** What is SaaS?

**Reference:** SaaS hosts software applications over the internet, allowing users to access and use them through web browsers without local installation.

**Generated:** A SQL is a query that combines rows from related tables.

- keyword recall: 0.0%, overlap: 0.0%, domain score: 15.0%

### 18k-eval-general_knowledge-001

**Q:** Predicted bounding boxes have low IoU with ground truth. What should you check first?

**Reference:** Verify the IoU threshold in your evaluation script. Lowering it to 0.3 may resolve discrepancies.

**Generated:** Underfitting choose actions from a branch of general patterns.

- keyword recall: 0.0%, overlap: 0.0%, domain score: 15.0%

### 18k-eval-general_knowledge-051

**Q:** How does a climate zone differ from a physical map?

**Reference:** Climate zones classify broad temperature and precipitation patterns, while physical maps visually depict these conditions through color gradients and contour lines.

**Generated:** A token is a token or numeric values.

- keyword recall: 0.0%, overlap: 0.0%, domain score: 15.0%

