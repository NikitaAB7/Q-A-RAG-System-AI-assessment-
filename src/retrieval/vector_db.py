# src/retrieval/vector_db.py

import chromadb
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self, db_path: str = "./chromadb"):
        logger.info(f"Initializing ChromaDB at {db_path}")
        
        # Use the new Chroma client
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None
    
    def create_collection(self, name: str = "compliance_docs"):
        """Create or get collection."""
        try:
            self.collection = self.client.get_collection(name)
            logger.info(f"✅ Using existing collection: {name}")
        except:
            self.collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Created new collection: {name}")
    
    def add_documents(self, chunks: List[Dict], embeddings_manager) -> None:
        """Index chunks into vector DB."""
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in chunks]
        
        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids, include=[])
            existing_ids = set(existing.get('ids', []))
        except Exception:
            existing_ids = set()
        
        new_chunks = [
            chunk for chunk, cid in zip(chunks, ids)
            if cid not in existing_ids
        ]
        
        if not new_chunks:
            logger.info("✅ All chunks already indexed. Skipping add.")
            return
        
        texts = [chunk['text'] for chunk in new_chunks]
        embeddings = embeddings_manager.embed_batch(texts)
        
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in new_chunks]
        metadatas = [
            {
                'doc_name': chunk['doc_name'],
                'page': str(chunk['page']),
                'chunk_id': str(chunk['chunk_id']),
                'source': chunk['source'],
                'section': chunk.get('section', ''),
                'subsection': chunk.get('subsection', '')
            }
            for chunk in new_chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"✅ Indexed {len(new_chunks)} chunks")
    
    def retrieve(self, query: str, embeddings_manager, 
                 top_k: int = 5,
                 metadata_filter: Dict = None) -> List[Dict]:
        """Retrieve top-k similar chunks."""
        query_embedding = embeddings_manager.embed_text(query)
        
        query_kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter
        
        results = self.collection.query(**query_kwargs)
        
        retrieved = []
        for i, (doc_id, distance, text, metadata) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['documents'][0],
            results['metadatas'][0]
        )):
            similarity_score = 1 - distance
            
            retrieved.append({
                'rank': i + 1,
                'chunk_id': metadata['chunk_id'],
                'doc_name': metadata['doc_name'],
                'page': int(metadata['page']),
                'text': text,
                'score': round(similarity_score, 3),
                'source': metadata['source'],
                'section': metadata.get('section', ''),
                'subsection': metadata.get('subsection', '')
            })
        
        return retrieved