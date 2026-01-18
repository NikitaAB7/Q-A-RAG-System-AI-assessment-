# Implementation Guide: Advanced RAG Features

## 1️⃣ HYBRID RETRIEVAL (Dense + BM25)

### Why Hybrid Retrieval?
- **Dense retrieval**: Good at semantic/conceptual similarity
- **Sparse retrieval (BM25)**: Excellent for exact keyword matches
- **Combined**: Gets best of both worlds (recall + precision)

### Architecture
```
Query
  ├─→ Dense Embedding → ChromaDB → Semantic Results
  ├─→ BM25 Index → Keyword Results
  └─→ Score Fusion → Normalized Scores → Merged Results
```

### Implementation Steps

#### Step 1: Install BM25 Library
```bash
pip install rank-bm25
```

#### Step 2: Create BM25 Index Builder
Create `src/retrieval/bm25_index.py`:

```python
# src/retrieval/bm25_index.py

from rank_bm25 import BM25Okapi
from typing import List, Dict
import pickle
import logging

logger = logging.getLogger(__name__)

class BM25IndexManager:
    def __init__(self, index_path: str = "./bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.chunks = []
        self.chunk_mapping = {}  # Maps chunk_id to chunk data
    
    def build_index(self, chunks: List[Dict]) -> None:
        """Build BM25 index from chunks."""
        self.chunks = chunks
        
        # Tokenize all chunks
        tokenized_chunks = []
        for chunk in chunks:
            # Simple tokenization: lowercase + split
            tokens = chunk['text'].lower().split()
            tokenized_chunks.append(tokens)
            self.chunk_mapping[chunk['chunk_id']] = chunk
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_chunks)
        logger.info(f"✅ Built BM25 index for {len(chunks)} chunks")
        
        # Save to disk
        self.save_index()
    
    def load_index(self) -> bool:
        """Load persisted BM25 index."""
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunk_mapping = data['mapping']
                logger.info(f"✅ Loaded BM25 index from {self.index_path}")
                return True
        except FileNotFoundError:
            logger.warning(f"⚠️ BM25 index not found at {self.index_path}")
            return False
    
    def save_index(self) -> None:
        """Persist BM25 index to disk."""
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'mapping': self.chunk_mapping
            }, f)
        logger.info(f"✅ Saved BM25 index to {self.index_path}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using BM25 scoring."""
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        # Build result list
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx]
            results.append({
                'rank': rank,
                'chunk_id': chunk['chunk_id'],
                'doc_name': chunk['doc_name'],
                'page': chunk['page'],
                'text': chunk['text'],
                'score': float(scores[idx]),
                'method': 'bm25'
            })
        
        return results
```

#### Step 3: Update Retriever for Hybrid Search
Update `src/retrieval/retriever.py`:

```python
# src/retrieval/retriever.py (ADD THIS METHOD)

from src.retrieval.bm25_index import BM25IndexManager
import numpy as np

class ComplianceRetriever:
    def __init__(self, vector_db, embeddings_manager, bm25_index_path: str = "./bm25_index.pkl"):
        self.vector_db = vector_db
        self.embeddings = embeddings_manager
        self.bm25_manager = BM25IndexManager(bm25_index_path)
        self.bm25_manager.load_index()
    
    def retrieve_hybrid(self, query: str, 
                       top_k: int = 5,
                       dense_weight: float = 0.5,
                       sparse_weight: float = 0.5,
                       score_threshold: float = 0.1) -> Dict:
        """Retrieve using hybrid (dense + sparse) approach."""
        
        # Get dense results
        dense_results = self.vector_db.retrieve(query, self.embeddings, top_k)
        dense_dict = {r['chunk_id']: r['score'] for r in dense_results}
        
        # Normalize dense scores [0, 1]
        dense_scores = list(dense_dict.values())
        if dense_scores:
            dense_min, dense_max = min(dense_scores), max(dense_scores)
            if dense_max > dense_min:
                dense_dict = {
                    cid: (score - dense_min) / (dense_max - dense_min)
                    for cid, score in dense_dict.items()
                }
        
        # Get sparse (BM25) results
        sparse_results = self.bm25_manager.retrieve(query, top_k)
        sparse_dict = {r['chunk_id']: r['score'] for r in sparse_results}
        
        # Normalize sparse scores [0, 1]
        sparse_scores = list(sparse_dict.values())
        if sparse_scores:
            sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
            if sparse_max > sparse_min:
                sparse_dict = {
                    cid: (score - sparse_min) / (sparse_max - sparse_min)
                    for cid, score in sparse_dict.items()
                }
        
        # Merge scores
        all_chunk_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
        merged_scores = {}
        
        for chunk_id in all_chunk_ids:
            dense_score = dense_dict.get(chunk_id, 0) * dense_weight
            sparse_score = sparse_dict.get(chunk_id, 0) * sparse_weight
            merged_scores[chunk_id] = dense_score + sparse_score
        
        # Sort and get top-k
        top_chunk_ids = sorted(
            merged_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build final results
        confident_chunks = []
        for chunk_id, score in top_chunk_ids:
            if score >= score_threshold:
                # Get chunk details from dense results
                chunk_data = next(
                    (r for r in dense_results if r['chunk_id'] == chunk_id),
                    None
                )
                if chunk_data:
                    chunk_data['hybrid_score'] = score
                    confident_chunks.append(chunk_data)
        
        avg_score = np.mean([c['hybrid_score'] for c in confident_chunks]) if confident_chunks else 0
        
        if avg_score > 0.7:
            confidence = 'high'
        elif avg_score > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'chunks': confident_chunks,
            'confidence': confidence,
            'avg_score': round(float(avg_score), 3),
            'total_retrieved': len(confident_chunks),
            'method': 'hybrid'
        }
```

#### Step 4: Update index.py to Build BM25
Add to `index.py`:

```python
# In index.py, after embedding/indexing chunks:

from src.retrieval.bm25_index import BM25IndexManager

print("\n[4/4] Building BM25 index...")
bm25_manager = BM25IndexManager()
bm25_manager.build_index(all_chunks)
```

#### Step 5: Update query.py to Use Hybrid Retrieval
```python
# In query.py, replace dense retrieval with:

retrieved = retriever.retrieve_hybrid(
    question, 
    top_k=5,
    dense_weight=0.6,  # Weight for semantic similarity
    sparse_weight=0.4,  # Weight for keyword matching
    score_threshold=0.1
)
```

---

## 2️⃣ RERANKING WITH CROSS-ENCODER

### Why Reranking?
- **Initial retrieval**: Fast but sometimes inaccurate
- **Cross-encoder reranking**: Precise scoring of query-chunk pairs
- **Result**: Better top-k chunks despite higher latency

### Architecture
```
Hybrid Retrieval (Top 50)
        ↓
    Cross-Encoder
    (Scores each pair)
        ↓
    Top-K After Reranking (5)
```

### Implementation Steps

#### Step 1: Install Cross-Encoder
```bash
pip install sentence-transformers
```

#### Step 2: Create Reranker Module
Create `src/retrieval/reranker.py`:

```python
# src/retrieval/reranker.py

from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/mmarco-MiniLMv2-L12-H384"):
        """
        Initialize cross-encoder reranker.
        
        Model options:
        - cross-encoder/mmarco-MiniLMv2-L12-H384 (Fast, lightweight)
        - cross-encoder/mmarco-MiniLMv2-L12-H384 (Recommended for financial)
        - cross-encoder/mmarco-MiniLM-L12-v2 (Fast, general)
        - cross-encoder/ms-marco-MiniLM-L-12-v2 (Production)
        """
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(self, query: str, 
               chunks: List[Dict],
               top_k: int = 5) -> List[Dict]:
        """Rerank chunks using cross-encoder."""
        
        if not chunks:
            return []
        
        # Prepare pairs: (query, chunk_text)
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Add scores to chunks and sort
        ranked_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = chunk.copy()
            chunk_copy['rerank_score'] = float(score)
            ranked_chunks.append(chunk_copy)
        
        # Sort by rerank score (descending)
        ranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Reranked {len(chunks)} chunks, top score: {ranked_chunks[0]['rerank_score']:.3f}")
        
        return ranked_chunks[:top_k]
```

#### Step 3: Update Retriever to Include Reranking
Add to `src/retrieval/retriever.py`:

```python
# Add to ComplianceRetriever class

from src.retrieval.reranker import CrossEncoderReranker

class ComplianceRetriever:
    def __init__(self, vector_db, embeddings_manager, 
                 bm25_index_path: str = "./bm25_index.pkl",
                 enable_reranking: bool = True):
        self.vector_db = vector_db
        self.embeddings = embeddings_manager
        self.bm25_manager = BM25IndexManager(bm25_index_path)
        self.bm25_manager.load_index()
        
        if enable_reranking:
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None
    
    def retrieve_with_reranking(self, query: str,
                               top_k: int = 5,
                               dense_weight: float = 0.6,
                               sparse_weight: float = 0.4,
                               score_threshold: float = 0.1,
                               rerank_top_k: int = 50) -> Dict:
        """Retrieve with hybrid search + cross-encoder reranking."""
        
        # Step 1: Hybrid retrieval (get top 50)
        initial_results = self.retrieve_hybrid(
            query,
            top_k=rerank_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            score_threshold=0.0  # Lower threshold for initial retrieval
        )
        
        chunks = initial_results['chunks']
        
        if not chunks:
            return initial_results
        
        # Step 2: Rerank with cross-encoder
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, top_k=top_k)
            confidence = 'reranked'
        
        # Step 3: Apply final threshold
        confident_chunks = [c for c in chunks if c.get('rerank_score', c.get('score', 0)) >= score_threshold]
        
        avg_score = sum(c.get('rerank_score', c.get('score', 0)) for c in confident_chunks) / len(confident_chunks) if confident_chunks else 0
        
        if avg_score > 0.7:
            confidence = 'high'
        elif avg_score > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'chunks': confident_chunks[:top_k],
            'confidence': confidence,
            'avg_score': round(float(avg_score), 3),
            'total_retrieved': len(confident_chunks),
            'method': 'hybrid_with_reranking'
        }
```

#### Step 4: Update query.py to Use Reranking
```python
# In query.py:

retrieved = retriever.retrieve_with_reranking(
    question,
    top_k=5,
    dense_weight=0.6,
    sparse_weight=0.4,
    score_threshold=0.1,
    rerank_top_k=50  # Initial retrieval pool before reranking
)
```

---

## 3️⃣ OPTIMIZED SEMANTIC CHUNKING

### Current Issues
- Chunks may be too large (semantic context lost)
- Chunks may be too small (lose continuity)
- No consideration for compliance document structure

### New Strategy
```
Financial Document
        ↓
    Extract Structure
    (Sections, Clauses)
        ↓
    Smart Boundaries
    (Regulation breaks, not just paragraphs)
        ↓
    Sized Chunks
    (300-500 tokens, with overlap)
        ↓
    Metadata Enrichment
    (Section, Subsection, Context)
```

### Implementation Steps

#### Step 1: Create Advanced Chunker
Create `src/indexing/advanced_chunker.py`:

```python
# src/indexing/advanced_chunker.py

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedChunker:
    """
    Smart chunking for financial/compliance documents.
    Respects regulatory structure while maintaining optimal chunk sizes.
    """
    
    def __init__(self, 
                 target_size: int = 400,        # Reduced from 600
                 min_chunk_size: int = 150,
                 overlap_tokens: int = 80,      # Increased overlap
                 respect_boundaries: bool = True):
        self.target_size = target_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
        self.respect_boundaries = respect_boundaries
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process all documents with advanced chunking."""
        all_chunks = []
        chunk_id = 0
        
        for doc in documents:
            text = doc.get('text', '')
            page_num = doc.get('page_num', 1)
            doc_name = doc.get('doc_name', 'Unknown')
            source = doc.get('source', 'Unknown')
            
            # Extract hierarchical structure
            sections = self._extract_regulatory_structure(text)
            
            for section_data in sections:
                section_text = section_data['content']
                section_meta = section_data['metadata']
                
                # Chunk each section
                section_chunks = self._smart_chunk_section(
                    section_text,
                    section_meta
                )
                
                for chunk_text in section_chunks:
                    if len(chunk_text.split()) >= self.min_chunk_size:
                        all_chunks.append({
                            'chunk_id': chunk_id,
                            'text': chunk_text.strip(),
                            'page': page_num,
                            'doc_name': doc_name,
                            'source': source,
                            'token_count': len(chunk_text.split()),
                            'section': section_meta.get('section', 'Unknown'),
                            'subsection': section_meta.get('subsection', ''),
                        })
                        chunk_id += 1
        
        logger.info(f"✅ Created {len(all_chunks)} advanced chunks from {len(documents)} pages")
        return all_chunks
    
    def _extract_regulatory_structure(self, text: str) -> List[Dict]:
        """Extract sections based on regulatory patterns."""
        sections = []
        
        # Patterns for financial regulations
        regulation_patterns = [
            (r'Regulation\s+(\d+[A-Z]*)', 'regulation'),
            (r'Schedule\s+([A-Z])', 'schedule'),
            (r'Section\s+(\d+)', 'section'),
            (r'(\d+\.\d+\.\d+)(?:\s|$)', 'subsection'),
            (r'Part\s+([A-Z])', 'part'),
        ]
        
        current_section = {'content': '', 'metadata': {}}
        
        lines = text.split('\n')
        for line in lines:
            matched = False
            
            for pattern, label in regulation_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Save current section
                    if current_section['content'].strip():
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'content': line + '\n',
                        'metadata': {label: match.group(1)}
                    }
                    matched = True
                    break
            
            if not matched:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections if sections else [{'content': text, 'metadata': {}}]
    
    def _smart_chunk_section(self, text: str, 
                            section_meta: Dict) -> List[str]:
        """Chunk a section smartly with overlap."""
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return [text] if len(text.split()) >= self.min_chunk_size else []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(para.split())
            
            # If adding paragraph exceeds target, save current chunk
            if current_tokens + para_tokens > self.target_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Overlap: keep last paragraph(s)
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_tokens = len(overlap_text.split())
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def calculate_statistics(self, chunks: List[Dict]) -> Dict:
        """Analyze chunk statistics."""
        token_counts = [c['token_count'] for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_chunk_size': min(token_counts) if token_counts else 0,
            'max_chunk_size': max(token_counts) if token_counts else 0,
            'chunks_by_section': self._group_by_section(chunks)
        }
    
    def _group_by_section(self, chunks: List[Dict]) -> Dict:
        """Group chunks by section."""
        grouped = {}
        for chunk in chunks:
            section = chunk.get('section', 'Unknown')
            grouped[section] = grouped.get(section, 0) + 1
        return grouped
```

#### Step 2: Update index.py to Use Advanced Chunker
```python
# In index.py:

from src.indexing.advanced_chunker import AdvancedChunker

print("\n[2/4] Chunking documents (advanced)...")
chunker = AdvancedChunker(
    target_size=400,      # Smaller for better relevance
    min_chunk_size=150,
    overlap_tokens=80
)
all_chunks = chunker.chunk_documents(all_documents)

# Show statistics
stats = chunker.calculate_statistics(all_chunks)
print(f"  Average chunk size: {stats['avg_chunk_size']:.0f} tokens")
print(f"  Range: {stats['min_chunk_size']}-{stats['max_chunk_size']} tokens")
```

#### Step 3: Update query.py to Display Enhanced Metadata
```python
# In query.py, when displaying chunks:

print("\nTop 3 chunks:")
for i, chunk in enumerate(retrieved['chunks'][:3], 1):
    section = chunk.get('section', 'N/A')
    subsection = chunk.get('subsection', '')
    location = f"{section}"
    if subsection:
        location += f" → {subsection}"
    
    print(f"  {i}. [{chunk['doc_name']}, Page {chunk['page']}, {location}]")
    print(f"     Score: {chunk.get('rerank_score', chunk.get('score')):.3f}")
    print(f"     {chunk['text'][:150]}...")
```

---

## 📋 INTEGRATION CHECKLIST

### Before Implementation
- [ ] Backup current `query.py` and `index.py`
- [ ] Test with sample questions
- [ ] Document baseline performance

### Phase 1: Hybrid Retrieval
- [ ] Install `rank-bm25`
- [ ] Create `src/retrieval/bm25_index.py`
- [ ] Add `retrieve_hybrid()` method to retriever
- [ ] Update `index.py` to build BM25 index
- [ ] Test with `query.py`
- [ ] Measure improvement (recall, precision)

### Phase 2: Reranking
- [ ] Install `sentence-transformers`
- [ ] Create `src/retrieval/reranker.py`
- [ ] Add `retrieve_with_reranking()` method
- [ ] Download cross-encoder model (first run)
- [ ] Update `query.py`
- [ ] Test latency impact
- [ ] Measure precision improvement

### Phase 3: Optimized Chunking
- [ ] Create `src/indexing/advanced_chunker.py`
- [ ] Update `index.py` to use new chunker
- [ ] Re-index documents
- [ ] Update citation display in `query.py`
- [ ] Compare retrieval quality vs old chunks

### Final Integration
- [ ] Update `config.yaml` with new parameters
- [ ] Add configuration for weights (dense/sparse)
- [ ] Add configuration for reranking model selection
- [ ] Update `requirements.txt`
- [ ] Document performance improvements in README

---

## 🚀 QUICK START (All 3 Features)

```bash
# 1. Install dependencies
pip install rank-bm25 sentence-transformers

# 2. Update files (follow guides above)

# 3. Re-index
python index.py

# 4. Test
python query.py
```

---

## ⚡ PERFORMANCE EXPECTATIONS

| Feature | Improvement | Latency Impact |
|---------|-------------|----------------|
| Hybrid Retrieval | +15-25% recall | +10-20ms |
| Reranking | +20-35% precision | +100-200ms |
| Optimized Chunking | +10-20% relevance | None (indexing only) |
| **All 3 Combined** | **+30-50% overall** | **+110-220ms** |

---

## 🔧 CONFIGURATION TEMPLATES

### config.yaml additions:

```yaml
# Retrieval
retrieval:
  method: "hybrid_with_reranking"    # Options: dense, hybrid, hybrid_with_reranking
  top_k: 5
  rerank_top_k: 50                   # Pool size before reranking
  
  # Hybrid weights
  dense_weight: 0.6
  sparse_weight: 0.4
  
  # Reranking
  enable_reranking: true
  reranker_model: "cross-encoder/mmarco-MiniLMv2-L12-H384"
  
  # BM25
  bm25_index_path: "./bm25_index.pkl"

# Chunking
chunking:
  strategy: "advanced_semantic"
  target_size: 400
  min_chunk_size: 150
  overlap_tokens: 80
  respect_boundaries: true
```

