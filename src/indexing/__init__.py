# src/indexing/__init__.py

from src.indexing.loader import load_documents
from src.indexing.chunking import chunk_by_semantics
from src.indexing.embeddings import EmbeddingManager

__all__ = ['load_documents', 'chunk_by_semantics', 'EmbeddingManager']