# src/retrieval/reranker.py

from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank chunks using cross-encoder."""
        if not chunks:
            return []
        
        pairs = [[query, chunk['text']] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        ranked_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = chunk.copy()
            chunk_copy['rerank_score'] = float(score)
            ranked_chunks.append(chunk_copy)
        
        ranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Reranked {len(chunks)} chunks, top score: {ranked_chunks[0]['rerank_score']:.3f}")
        return ranked_chunks[:top_k]
