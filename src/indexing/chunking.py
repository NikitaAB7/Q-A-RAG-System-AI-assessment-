# src/indexing/chunking.py

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self, target_size: int = 600, 
                 overlap_tokens: int = 100,
                 min_chunk_size: int = 150):
        self.target_size = target_size
        self.overlap_tokens = overlap_tokens
        self.min_chunk_size = min_chunk_size
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process all documents and create chunks."""
        all_chunks = []
        chunk_id = 0
        
        for doc in documents:
            text = doc.get('text', '')
            
            # Split by semantic boundaries
            sections = self._split_by_headers(text)
            
            for section in sections:
                # Further split if too long
                if len(section.split()) > self.target_size:
                    sub_chunks = self._split_by_paragraphs(section)
                else:
                    sub_chunks = [section]
                
                for sub_chunk in sub_chunks:
                    if len(sub_chunk.split()) >= self.min_chunk_size:
                        all_chunks.append({
                            'chunk_id': chunk_id,
                            'text': sub_chunk.strip(),
                            'page': doc['page_num'],
                            'doc_name': doc['doc_name'],
                            'source': doc['source'],
                            'token_count': len(sub_chunk.split())
                        })
                        chunk_id += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} pages")
        return all_chunks
    
    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by regulatory headers."""
        pattern = r'(?=Regulation\s+\d+|Section\s+\d+|\d+\.\d+\.\d+|\n[A-Z][A-Z\s]+:)'
        sections = re.split(pattern, text)
        return [s.strip() for s in sections if s.strip()]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split by paragraphs with overlap."""
        paragraphs = text.split('\n\n')
        chunks = []
        current = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            
            if current_size + para_size > self.target_size and current:
                chunks.append('\n\n'.join(current))
                current = current[-1:] if current else []
                current_size = len(current[-1].split()) if current else 0
            
            current.append(para)
            current_size += para_size
        
        if current:
            chunks.append('\n\n'.join(current))
        
        return chunks

def chunk_by_semantics(documents: List[Dict], 
                       target_size: int = 600,
                       overlap_tokens: int = 100) -> List[Dict]:
    """Convenience function."""
    chunker = Chunker(target_size, overlap_tokens)
    return chunker.chunk_documents(documents)