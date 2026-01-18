# src/retrieval/retriever.py

from typing import Dict, List
import logging
import numpy as np
from src.retrieval.bm25_index import BM25IndexManager
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

class ComplianceRetriever:
    def __init__(self, vector_db, embeddings_manager,
                 bm25_index_path: str = "./bm25_index.pkl",
                 enable_reranking: bool = True):
        self.vector_db = vector_db
        self.embeddings = embeddings_manager
        self.bm25_manager = BM25IndexManager(bm25_index_path)
        self.bm25_manager.load_index()
        self.reranker = CrossEncoderReranker() if enable_reranking else None
    
    def retrieve_with_confidence(self, query: str, 
                                 top_k: int = 5,
                                 score_threshold: float = 0.1) -> Dict:
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

    def retrieve_hybrid(self, query: str,
                        top_k: int = 5,
                        dense_weight: float = 0.6,
                        sparse_weight: float = 0.4,
                        score_threshold: float = 0.1) -> Dict:
        """Retrieve using hybrid (dense + BM25) approach."""

        dense_results = self.vector_db.retrieve(query, self.embeddings, top_k)
        dense_dict = {str(r['chunk_id']): r['score'] for r in dense_results}

        dense_scores = list(dense_dict.values())
        if dense_scores:
            dense_min, dense_max = min(dense_scores), max(dense_scores)
            if dense_max > dense_min:
                dense_dict = {
                    cid: (score - dense_min) / (dense_max - dense_min)
                    for cid, score in dense_dict.items()
                }

        sparse_results = self.bm25_manager.retrieve(query, top_k)
        sparse_dict = {str(r['chunk_id']): r['score'] for r in sparse_results}

        sparse_scores = list(sparse_dict.values())
        if sparse_scores:
            sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
            if sparse_max > sparse_min:
                sparse_dict = {
                    cid: (score - sparse_min) / (sparse_max - sparse_min)
                    for cid, score in sparse_dict.items()
                }

        all_chunk_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
        merged_scores = {}

        for chunk_id in all_chunk_ids:
            dense_score = dense_dict.get(chunk_id, 0) * dense_weight
            sparse_score = sparse_dict.get(chunk_id, 0) * sparse_weight
            merged_scores[chunk_id] = dense_score + sparse_score

        top_chunk_ids = sorted(
            merged_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        confident_chunks = []
        for chunk_id, score in top_chunk_ids:
            if score >= score_threshold:
                chunk_data = next(
                    (r for r in dense_results if str(r['chunk_id']) == chunk_id),
                    None
                )
                if chunk_data:
                    chunk_data['hybrid_score'] = score
                    confident_chunks.append(chunk_data)

        avg_score = float(np.mean([c['hybrid_score'] for c in confident_chunks])) if confident_chunks else 0

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
            'total_retrieved': len(confident_chunks),
            'method': 'hybrid'
        }

    def retrieve_with_reranking(self, query: str,
                                top_k: int = 5,
                                dense_weight: float = 0.6,
                                sparse_weight: float = 0.4,
                                score_threshold: float = 0.1,
                                rerank_top_k: int = 50) -> Dict:
        """Retrieve with hybrid search + cross-encoder reranking."""

        initial_results = self.retrieve_hybrid(
            query,
            top_k=rerank_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            score_threshold=0.0
        )

        chunks = initial_results['chunks']
        if not chunks:
            return initial_results

        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, top_k=top_k)

        confident_chunks = [
            c for c in chunks
            if c.get('rerank_score', c.get('hybrid_score', c.get('score', 0))) >= score_threshold
        ]

        avg_score = (
            sum(c.get('rerank_score', c.get('hybrid_score', c.get('score', 0))) for c in confident_chunks) /
            len(confident_chunks)
        ) if confident_chunks else 0

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