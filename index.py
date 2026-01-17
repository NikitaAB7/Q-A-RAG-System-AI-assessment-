# index.py

import yaml
import logging
from pathlib import Path
from src.indexing.loader import load_documents
from src.indexing.chunking import chunk_by_semantics
from src.indexing.embeddings import EmbeddingManager
from src.retrieval.vector_db import VectorDB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("COMPLIANCE RAG - INDEXING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Load documents
    logger.info("\n[1/4] Loading PDF documents...")
    documents = load_documents("./docs")
    
    if not documents:
        logger.error("No documents found. Please add PDFs to ./docs folder")
        return
    
    logger.info(f"Loaded {len(documents)} pages from documents")
    
    # Step 2: Chunk documents
    logger.info("\n[2/4] Chunking documents...")
    chunks = chunk_by_semantics(documents, target_size=600, overlap_tokens=100)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3: Initialize embeddings and vector DB
    logger.info("\n[3/4] Initializing embeddings model...")
    embeddings_manager = EmbeddingManager(model_name="all-mpnet-base-v2", device="cpu")
    
    logger.info("\n[3.5/4] Initializing vector database...")
    vector_db = VectorDB(db_path="./chromadb")
    vector_db.create_collection("compliance_docs")
    
    # Step 4: Embed and index
    logger.info("\n[4/4] Embedding and indexing chunks...")
    vector_db.add_documents(chunks, embeddings_manager)
    
    logger.info("\n" + "="*80)
    logger.info("INDEXING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total documents indexed: {len(documents)} pages")
    logger.info(f"Total chunks created: {len(chunks)}")
    logger.info(f"Vector DB: ./chromadb")
    logger.info(f"Collection: compliance_docs")
    logger.info("\nYou can now run queries using the RAG system!")

if __name__ == "__main__":
    main()