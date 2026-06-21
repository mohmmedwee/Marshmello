# Phase 18K — General Benchmark (base)

- Generated: 2026-06-21T10:22:11.181806+00:00
- Checkpoint: `/Users/ehabshobaki/Documents/myproject/learning/mini-transformer-from-scratch/13_gpt_pretraining/checkpoints/large_50m/latest.pt`
- Questions: 500
- Decoding: greedy

## Overall metrics

| Metric | Value |
|---|---:|
| Keyword recall | 3.0% |
| Reference token overlap | 9.9% |
| Response length sanity | 99.4% |
| Repetition rate | 67.6% |
| Empty response rate | 0.4% |
| Hallucination proxy | 84.2% |
| Domain score (composite) | 11.7% |

## By benchmark bucket

| Bucket | Domain score | Keyword | Overlap | Hallucination |
|---|---:|---:|---:|---:|
| ai | 13.6% | 3.7% | 10.4% | 79.0% |
| databases | 14.7% | 4.9% | 10.9% | 79.0% |
| general_knowledge | 7.8% | 2.0% | 9.6% | 92.0% |
| programming | 11.9% | 2.0% | 9.8% | 83.0% |
| system_design | 10.3% | 2.5% | 9.0% | 88.0% |

## Sample outputs (10)

### 18k-eval-ai-001

**Q:** Explain how an encoder handles syntactic features.

**Reference:** The encoder in a text classification model transforms input sentences into a fixed-length vector representation. For instance, a transformer's encoder captures syntactic and semantic features of the input, enabling accurate sentiment classification even with varying sentence structures.

**Generated:** An example of a type of a type of text is the word "artificial intelligence" is a text that uses a text to the context of a text and generate a text that uses a text to generate a prompt.

- keyword recall: 10.0%, overlap: 20.5%, domain score: 10.7%

### 18k-eval-ai-051

**Q:** How is a number line different from a coordinate system?

**Reference:** A number line is a 1D visual representation of numbers, while a coordinate system extends this to 2D or more dimensions. Both use numbers to locate points, but coordinate systems handle multiple axes.

**Generated:** A group of different types of different types of different types of types of types of types of types of weights. These types of types of types of types of types of types of types of weightning types of weightning types of weightning and weightning types of weightning behavior and weights.

- keyword recall: 0.0%, overlap: 4.8%, domain score: 1.7%

### 18k-eval-databases-001

**Q:** How does a primary key function within a database table?

**Reference:** A primary key uniquely identifies each row by enforcing uniqueness and preventing duplicate entries, allowing efficient data retrieval and relationships with foreign keys.

**Generated:** A database is a field that stores records as a set of instructions that guide a model toward the desired answer. It helps the system answer or act more usefully.

- keyword recall: 0.0%, overlap: 3.8%, domain score: 16.3%

### 18k-eval-databases-051

**Q:** Define ACID properties in database systems.

**Reference:** In database systems, ACID properties ensure reliable transaction processing through Atomicity (all-or-nothing execution), Consistency (valid state transitions), Isolation (concurrent transactions' independence), and Durability (persistent post-commit data). These principles guarantee robustness in relational and transactional databases.

**Generated:** CID is a type of database that stores records as a database structure. It helps keep data useful, reliable, or fast to access.

- keyword recall: 20.0%, overlap: 13.1%, domain score: 41.6%

### 18k-eval-programming-001

**Q:** How does the content type header differ from the caching header?

**Reference:** The content type header specifies the data format (e.g., JSON, HTML), while the caching header determines how long a response is stored in the browser.

**Generated:** The main difference between a web browser and a web browser is a web browser that is a web browser and a web browser to run a web server. The web browser is a web server is a server that provides a web server to run the web server.

- keyword recall: 0.0%, overlap: 16.0%, domain score: 5.6%

### 18k-eval-programming-051

**Q:** How does hash map lookup function?

**Reference:** A hash map lookup efficiently finds a value by computing the key's hash and accessing the corresponding stored value.

**Generated:** Function is a reusable block of code that can be changed. It helps Python code stay clear and practical.

- keyword recall: 0.0%, overlap: 10.5%, domain score: 18.7%

### 18k-eval-system_design-001

**Q:** Define 'signal' in Linux.

**Reference:** A signal in Linux is a software interrupt used to notify a process of an event. Common signals include SIGKILL and SIGTERM, which trigger specific actions like termination or suspension.

**Generated:** The term 'unsupervised learning is finding structure in data without labeled data without labels. It is useful for building models that work beyond one example.

- keyword recall: 0.0%, overlap: 7.3%, domain score: 17.5%

### 18k-eval-system_design-051

**Q:** What is SaaS?

**Reference:** SaaS hosts software applications over the internet, allowing users to access and use them through web browsers without local installation.

**Generated:** Sft is a text format for a query results. It helps keep data useful, reliable, or fast to access.

- keyword recall: 10.0%, overlap: 10.3%, domain score: 22.1%

### 18k-eval-general_knowledge-001

**Q:** Predicted bounding boxes have low IoU with ground truth. What should you check first?

**Reference:** Verify the IoU threshold in your evaluation script. Lowering it to 0.3 may resolve discrepancies.

**Generated:** To make a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack of a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a pack for a p

- keyword recall: 0.0%, overlap: 1.9%, domain score: 0.7%

### 18k-eval-general_knowledge-051

**Q:** How does a climate zone differ from a physical map?

**Reference:** Climate zones classify broad temperature and precipitation patterns, while physical maps visually depict these conditions through color gradients and contour lines.

**Generated:** A good idea for a person should have a person who is important to take their own life. This should have a positive impact on their own life. It should also be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able

- keyword recall: 0.0%, overlap: 0.0%, domain score: 0.0%

