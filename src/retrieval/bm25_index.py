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
        self.chunks: List[Dict] = []
        self.chunk_mapping: Dict[str, Dict] = {}
    
    def build_index(self, chunks: List[Dict]) -> None:
        """Build BM25 index from chunks and persist it."""
        self.chunks = chunks
        self.chunk_mapping = {}
        
        tokenized_chunks = []
        for chunk in chunks:
            tokens = chunk['text'].lower().split()
            tokenized_chunks.append(tokens)
            self.chunk_mapping[str(chunk['chunk_id'])] = chunk
        
        self.bm25 = BM25Okapi(tokenized_chunks)
        logger.info(f"✅ Built BM25 index for {len(chunks)} chunks")
        self.save_index()
    
    def load_index(self) -> bool:
        """Load persisted BM25 index."""
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunk_mapping = data['mapping']
                self.chunks = data['chunks']
                logger.info(f"✅ Loaded BM25 index from {self.index_path}")
                return True
        except FileNotFoundError:
            logger.warning(f"⚠️ BM25 index not found at {self.index_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def save_index(self) -> None:
        """Persist BM25 index to disk."""
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'mapping': self.chunk_mapping,
                'chunks': self.chunks
            }, f)
        logger.info(f"✅ Saved BM25 index to {self.index_path}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using BM25 scoring."""
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index not initialized")
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx]
            results.append({
                'rank': rank,
                'chunk_id': str(chunk['chunk_id']),
                'doc_name': chunk['doc_name'],
                'page': chunk['page'],
                'text': chunk['text'],
                'score': float(scores[idx]),
                'source': chunk.get('source'),
                'method': 'bm25'
            })
        
        return results
