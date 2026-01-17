# src/indexing/embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name: str = "all-mpnet-base-v2", 
                 device: str = "cpu"):
        logger.info(f"Loading embedding model: {model_name} on device: {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except AssertionError:
            # Fallback to CPU if CUDA is not available
            logger.warning(f"CUDA not available, falling back to CPU")
            self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently."""
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings