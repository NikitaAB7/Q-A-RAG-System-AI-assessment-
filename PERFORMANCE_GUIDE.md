# Performance Improvement Guide

## Current Issues & Solutions

### 1. ❌ NO CHUNKS BEING RETRIEVED (Critical)
**Problem:** `avg_retrieved_chunks: 0.0`  
**Root Cause:** Retrieval threshold `0.3` is too high - all similarity scores below threshold  
**Solution:** ✅ **IMPLEMENTED** - Lowered threshold to `0.1`

**Changes Made:**
- `evaluation_metrics.py`: Line 113 - Changed threshold from 0.3 to 0.1
- `query.py`: Line 30 - Changed threshold from 0.3 to 0.1

**To manually adjust:**
```python
# In .env or config
RETRIEVAL_SCORE_THRESHOLD=0.1  # Start low, increase if too many false positives
```

---

## Performance Improvement Checklist

### Phase 1: Quick Wins (Do These First)
- [x] Lower retrieval threshold (0.3 → 0.1)
- [ ] Switch to faster embedding model
- [ ] Reduce chunk size for better relevance
- [ ] Test with real queries

### Phase 2: Medium Effort
- [ ] Fine-tune embedding model on domain data
- [ ] Optimize chunking strategy
- [ ] Add query preprocessing
- [ ] Implement query expansion

### Phase 3: Advanced
- [ ] Hybrid retrieval (dense + BM25)
- [ ] Re-ranking with cross-encoder
- [ ] Semantic cache
- [ ] Query optimization

---

## Detailed Solutions

### 1. Embedding Model Optimization

**Current:** `all-mpnet-base-v2` (good general model)

**For Financial/Compliance, try:**

```bash
ollama pull llama2  # Multi-domain
# OR download specialized model:
pip install sentence-transformers
```

**Better options for SEBI compliance:**
```python
# src/indexing/embeddings.py
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):  # Faster, good quality
        # "all-MiniLM-L6-v2" - faster (good for RAG)
        # "all-mpnet-base-v2" - best quality (current)
        # "paraphrase-MiniLM-L6-v2" - good for paraphrasing
```

### 2. Chunking Strategy

**Current:** 600 tokens/chunk with semantic splitting

**Improved strategy:**
```python
# src/indexing/chunking.py
def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
    chunker = Chunker(
        target_size=300,        # REDUCE from 600 (faster retrieval)
        overlap_tokens=50,      # REDUCE from 100 (less redundancy)
        min_chunk_size=75       # REDUCE from 150
    )
```

**Why this helps:**
- Smaller chunks = more precise retrieval
- Faster to embed and search
- Better relevance matching

**Command to re-index:**
```bash
python index.py
```

### 3. Query Preprocessing

**Add query expansion:**
```python
# query.py
def query_rag_enhanced(question: str):
    # Expand question with synonyms
    expanded_query = f"""
    Question: {question}
    
    Related terms:
    - Equity derivatives exposure
    - Mutual fund regulations
    - SEBI compliance
    """
    retrieved = retriever.retrieve_with_confidence(expanded_query, top_k=5)
```

### 4. Retrieval Threshold Tuning

**Current:** 0.1 (after fix)

**Adaptive threshold:**
```python
# If low recall, lower threshold
THRESHOLD_MAPPING = {
    'high_precision': 0.5,   # Return only best matches
    'balanced': 0.25,        # Default
    'high_recall': 0.1,      # Return all possible matches
}
```

### 5. Switch to Faster Embedding Model

**If latency is still high:**
```bash
# Faster model (better for real-time)
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Update `src/indexing/embeddings.py`:
```python
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        # Faster model = 2-3x speedup
        self.model = SentenceTransformer(model_name, device=device)
```

Then re-index:
```bash
python index.py
```

### 6. Implement Hybrid Retrieval

**Combine dense (semantic) + sparse (BM25):**
```python
# src/retrieval/retriever.py
class ComplianceRetriever:
    def retrieve_hybrid(self, query: str, top_k: int = 5):
        # Dense retrieval (current)
        dense_results = self.retrieve_dense(query, top_k)
        
        # Sparse retrieval (BM25)
        sparse_results = self.retrieve_bm25(query, top_k)
        
        # Combine and re-rank
        combined = self._merge_results(dense_results, sparse_results)
        return combined[:top_k]
```

---

## Expected Improvements

### After Phase 1 (Threshold Fix)
```
Before → After:
- Chunks Retrieved: 0 → 4-5 ✅
- Faithfulness: 0.2 → 0.6+ ✅
- Citation Rate: 0% → 70%+ ✅
- Latency: 0.1s → 1-2s (acceptable)
```

### After Phase 2 (Chunking Optimization)
```
- Retrieval Quality: +30% better matches
- Latency: 1-2s → 0.5-1s
- Storage: -50% smaller index
```

### After Phase 3 (Hybrid Retrieval)
```
- Recall@5: 85%+ (currently 0%)
- Faithfulness: 0.8+ (currently 0.2)
- Citation Accuracy: 90%+ (currently 0%)
```

---

## Quick Start - Execute These Steps

### Step 1: Test with lowered threshold
```bash
python evaluation_metrics.py
# Check if avg_retrieved_chunks > 0
```

### Step 2: If still 0 chunks, debug retrieval
```python
from query import query_rag
result = query_rag("What is the maximum exposure limit for equity derivatives?")
# Check if "Retrieved 5 chunks" appears
```

### Step 3: Try smaller chunks
Edit `index.py`:
```python
chunks = chunk_by_semantics(documents, target_size=300, overlap_tokens=50)
```
Then run:
```bash
python index.py
python evaluation_metrics.py
```

### Step 4: Switch embedding model (if needed)
Edit `src/indexing/embeddings.py`:
```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
```
Then:
```bash
python index.py
python evaluation_metrics.py
```

---

## Monitoring Improvements

Track these metrics:

```python
# In evaluation_metrics.py
METRICS = {
    'avg_retrieved_chunks': 'Should be 5',
    'retrieval_recall_at_5': 'Should be 0.8+',
    'faithfulness_score': 'Should be 0.7+',
    'citation_rate': 'Should be 0.9+',
    'avg_latency_seconds': 'Should be 1-3s'
}
```

---

## Troubleshooting

### Still no chunks after threshold fix?
```python
# Check similarity scores manually
from src.retrieval.vector_db import VectorDB
from src.indexing.embeddings import EmbeddingManager

vector_db = VectorDB()
vector_db.create_collection("compliance_docs")
embeddings = EmbeddingManager()

query = "What is the maximum exposure limit?"
results = vector_db.retrieve(query, embeddings, top_k=5)

for r in results:
    print(f"Score: {r['score']:.3f} - {r['text'][:100]}")
# If all scores < 0.1, try even lower threshold or re-index with different chunking
```

### Latency too high after improvement?
```bash
# Switch to MiniLM (faster model)
# Update model in embeddings.py to "all-MiniLM-L6-v2"
# Re-index: python index.py
# Re-evaluate: python evaluation_metrics.py
```

---

## Summary

| Change | Impact | Effort | Time |
|--------|--------|--------|------|
| Lower threshold (0.3→0.1) | Critical ✅ | 5 min | Done |
| Re-chunk (600→300 tokens) | High | 15 min | ~3 min |
| Faster embedding model | Medium | 10 min | ~3 min |
| Hybrid retrieval | High | 30 min | ~2 min search |

**Recommended Priority:**
1. ✅ Lower threshold (already done)
2. Re-index with smaller chunks
3. Switch embedding model
4. Test with real queries
5. Implement hybrid retrieval (if needed)

