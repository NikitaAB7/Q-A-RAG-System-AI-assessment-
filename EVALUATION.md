# RAG Evaluation Framework

## Overview
This evaluation framework measures three key metrics for RAG (Retrieval-Augmented Generation) systems:

---

## Metrics Explained

### 1. RETRIEVAL RECALL@5
**What it measures:** Whether relevant documents are being retrieved

**Formula:**
```
Recall@5 = (Relevant docs in top-5) / (Total relevant docs)
```

**Interpretation:**
- **1.0** = Perfect retrieval - all relevant documents found in top-5
- **0.5** = 50% of relevant docs retrieved
- **0.0** = No relevant docs retrieved

**Why it matters:** A RAG system is only as good as its retrieval. If relevant context isn't retrieved, the LLM can't use it.

---

### 2. FAITHFULNESS (LLM-as-Judge)
**What it measures:** Whether the generated answer is grounded in the retrieved context

**Method:** Uses Mistral as a judge to rate answer faithfulness on a 1-5 scale:
- **5** = Completely faithful & accurate
- **4** = Mostly faithful, minor inaccuracies
- **3** = Partially faithful, some inaccuracies
- **2** = Mostly unfaithful
- **1** = Contradicts context

**Conversion to score:** `faithfulness_score = rating / 5.0` (0.0 to 1.0)

**Why it matters:** Ensures the answer is grounded in provided documents, not hallucinated.

---

### 3. CITATION CORRECTNESS
**What it measures:** Quality and quantity of citations in answers

**Sub-metrics:**
- **Citation Rate:** % of answers with citations (should be 100%)
- **Citation Count:** Number of sources cited per answer
- **Citation Format:** Whether citations follow proper format `[doc, page, chunk]`
- **Citation Precision:** Ratio of cited chunks to available chunks

**Why it matters:** Citations allow users to verify answers and trace source information.

---

## Running Evaluation

### Basic Usage
```bash
python evaluation_metrics.py
```

### Output Files
- `evaluation_results.json` - Detailed results for each question and aggregated metrics

### Sample Output
```
=== EVALUATION SUMMARY ===
Questions evaluated: 20

--- METRIC 1: RETRIEVAL RECALL@5 ---
Avg Recall@5: 0.850
(Higher is better: 1.0 = all relevant docs retrieved)

--- METRIC 2: FAITHFULNESS (LLM-as-Judge) ---
Avg Faithfulness: 0.720
(Scale 0-1: measures if answer is grounded in context)

--- METRIC 3: CITATION CORRECTNESS ---
Citation Rate: 95.0%
Avg Chunks Retrieved: 4.5
(Checks if answers include proper citations)

--- PERFORMANCE ---
Avg Latency: 2.350s
```

---

## Interpreting Results

### Good RAG System
- Recall@5: > 0.80 (retrieves most relevant docs)
- Faithfulness: > 0.70 (answers grounded in context)
- Citation Rate: > 0.90 (most answers cited)
- Latency: < 5s (fast enough for interactive use)

### Areas to Improve
- **Low Recall** → Better retrieval model or chunking strategy
- **Low Faithfulness** → Adjust system prompt or use better LLM
- **Low Citation Rate** → Modify generation prompt to require citations

---

## Implementation Details

### Metric 1: Recall Calculation
- For each question, retrieves top-5 chunks
- Compares retrieved document names with ground truth
- Calculates intersection / union ratio

### Metric 2: Faithfulness Evaluation
- Sends answer + context to Mistral
- Mistral acts as judge, rates faithfulness 1-5
- Results aggregated for overall faithfulness score

### Metric 3: Citation Analysis
- Parses answer for `[source]` patterns
- Validates citation format
- Counts citations vs retrieved chunks

---

## Questions Dataset

File: `questions.json`
- Contains 20 SEBI compliance questions on derivatives
- Format:
```json
[
  {"id": 1, "question": "What is the maximum exposure limit..."},
  {"id": 2, "question": "Are index options considered..."},
  ...
]
```

---

## Components Used

- **Retriever:** `ComplianceRetriever` (semantic search with confidence scoring)
- **Generator:** `GroundedAnswerGenerator` (Mistral 7B via Ollama)
- **Embeddings:** `EmbeddingManager` (all-mpnet-base-v2, CPU-based)
- **Vector DB:** ChromaDB (persistent storage)

---

## Customization

### Change Number of Questions
```python
evaluator.evaluate_rag(questions, num_questions=50)
```

### Change Retrieval Top-K
Edit in `evaluate_faithfulness()`:
```python
retrieved = self.retriever.retrieve_with_confidence(question, top_k=10)
```

### Adjust Faithfulness Criteria
Modify `faithfulness_prompt` in `evaluate_faithfulness()` method.

---

## Notes

- Evaluation uses local Mistral 7B (no API calls)
- Results depend on indexed documents (empty DB = all low metrics)
- First run will download embedding models (~400MB)
- Takes ~2-5 seconds per question depending on hardware

