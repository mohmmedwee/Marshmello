# Phase 18K — General Benchmark (broad_sft_latest)

- Generated: 2026-06-21T10:36:59.386654+00:00
- Checkpoint: `18B_marshmello_instruct/checkpoints/latest.pt`
- Questions: 500
- Decoding: greedy

## Overall metrics

| Metric | Value |
|---|---:|
| Keyword recall | 2.3% |
| Reference token overlap | 9.6% |
| Response length sanity | 97.8% |
| Repetition rate | 63.4% |
| Empty response rate | 2.2% |
| Hallucination proxy | 82.2% |
| Domain score (composite) | 12.0% |

## By benchmark bucket

| Bucket | Domain score | Keyword | Overlap | Hallucination |
|---|---:|---:|---:|---:|
| ai | 13.9% | 4.1% | 10.7% | 81.0% |
| databases | 13.4% | 4.2% | 10.6% | 77.0% |
| general_knowledge | 12.4% | 1.4% | 10.4% | 84.0% |
| programming | 10.8% | 0.7% | 7.4% | 83.0% |
| system_design | 9.5% | 1.2% | 8.9% | 86.0% |

## Sample outputs (10)

### 18k-eval-ai-001

**Q:** Explain how an encoder handles syntactic features.

**Reference:** The encoder in a text classification model transforms input sentences into a fixed-length vector representation. For instance, a transformer's encoder captures syntactic and semantic features of the input, enabling accurate sentiment classification even with varying sentence structures.

**Generated:** An abstraction is a method that combines data into training, and other layers to make predictions based on new data.

- keyword recall: 0.0%, overlap: 10.3%, domain score: 18.6%

### 18k-eval-ai-051

**Q:** How is a number line different from a coordinate system?

**Reference:** A number line is a 1D visual representation of numbers, while a coordinate system extends this to 2D or more dimensions. Both use numbers to locate points, but coordinate systems handle multiple axes.

**Generated:** A primary key is a set of column or a set of column set that a set of a set of a set of set of a set of set of a set of set of a set of set of a set of set of a set of set is a set of set of a set of set of set of a set of set of a set of set of a set of set of set of a set of set of a set of a set of set of set of a training set of set of set of a training a training choices that guide a model can then guide a model

- keyword recall: 0.0%, overlap: 8.0%, domain score: 2.8%

### 18k-eval-databases-001

**Q:** How does a primary key function within a database table?

**Reference:** A primary key uniquely identifies each row by enforcing uniqueness and preventing duplicate entries, allowing efficient data retrieval and relationships with foreign keys.

**Generated:** A database is a column set of column column column each row each row in a table.

- keyword recall: 10.0%, overlap: 15.0%, domain score: 38.7%

### 18k-eval-databases-051

**Q:** Define ACID properties in database systems.

**Reference:** In database systems, ACID properties ensure reliable transaction processing through Atomicity (all-or-nothing execution), Consistency (valid state transitions), Isolation (concurrent transactions' independence), and Durability (persistent post-commit data). These principles guarantee robustness in relational and transactional databases.

**Generated:** AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

- keyword recall: 0.0%, overlap: 0.0%, domain score: 0.0%

### 18k-eval-programming-001

**Q:** How does the content type header differ from the caching header?

**Reference:** The content type header specifies the data format (e.g., JSON, HTML), while the caching header determines how long a response is stored in the browser.

**Generated:** The primary key features a neural network architecture that combines with each layers.

- keyword recall: 0.0%, overlap: 10.3%, domain score: 18.6%

### 18k-eval-programming-051

**Q:** How does hash map lookup function?

**Reference:** A hash map lookup efficiently finds a value by computing the key's hash and accessing the corresponding stored value.

**Generated:** Fine-tuning occurs when the hallucination is the maximum number of column number of vector for a number of vector or infinite number of vector or infinite number of vector or infinite number of vector or inosion.

- keyword recall: 0.0%, overlap: 10.7%, domain score: 3.8%

### 18k-eval-system_design-001

**Q:** Define 'signal' in Linux.

**Reference:** A signal in Linux is a software interrupt used to notify a process of an event. Common signals include SIGKILL and SIGTERM, which trigger specific actions like termination or suspension.

**Generated:** The term 'unknown as a type of used in the same sequence of interaction in the same value.

- keyword recall: 0.0%, overlap: 16.7%, domain score: 5.8%

### 18k-eval-system_design-051

**Q:** What is SaaS?

**Reference:** SaaS hosts software applications over the internet, allowing users to access and use them through web browsers without local installation.

**Generated:** NER is the SQL query to read data.

- keyword recall: 0.0%, overlap: 14.3%, domain score: 35.0%

### 18k-eval-general_knowledge-001

**Q:** Predicted bounding boxes have low IoU with ground truth. What should you check first?

**Reference:** Verify the IoU threshold in your evaluation script. Lowering it to 0.3 may resolve discrepancies.

**Generated:** To make a matching a matching or a matching context.

- keyword recall: 0.0%, overlap: 7.7%, domain score: 17.7%

### 18k-eval-general_knowledge-051

**Q:** How does a climate zone differ from a physical map?

**Reference:** Climate zones classify broad temperature and precipitation patterns, while physical maps visually depict these conditions through color gradients and contour lines.

**Generated:** A reusable reusable blocks of the other hand, are a full of reusable blocks that can be used to a query.

- keyword recall: 0.0%, overlap: 0.0%, domain score: 15.0%

