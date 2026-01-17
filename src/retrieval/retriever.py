# src/retrieval/retriever.py

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ComplianceRetriever:
    def __init__(self, vector_db, embeddings_manager):
        self.vector_db = vector_db
        self.embeddings = embeddings_manager
    
    def retrieve_with_confidence(self, query: str, 
                                 top_k: int = 5,
                                 score_threshold: float = 0.3) -> Dict:
        """Retrieve chunks with confidence metrics."""
        
        chunks = self.vector_db.retrieve(query, self.embeddings, top_k)
        
        confident_chunks = [c for c in chunks if c['score'] >= score_threshold]
        
        if not confident_chunks:
            return {
                'chunks': [],
                'confidence': 'low',
                'avg_score': 0,
                'message': f'No results above threshold ({score_threshold})'
            }
        
        avg_score = sum(c['score'] for c in confident_chunks) / len(confident_chunks)
        
        if avg_score > 0.7:
            confidence = 'high'
        elif avg_score > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'chunks': confident_chunks,
            'confidence': confidence,
            'avg_score': round(avg_score, 3),
            'total_retrieved': len(confident_chunks)
        }