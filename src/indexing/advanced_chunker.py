# src/indexing/advanced_chunker.py

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class AdvancedChunker:
    """Smart chunking for financial/compliance documents."""
    
    def __init__(self, 
                 target_size: int = 400,
                 min_chunk_size: int = 150,
                 overlap_tokens: int = 80,
                 respect_boundaries: bool = True):
        self.target_size = target_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
        self.respect_boundaries = respect_boundaries
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []
        chunk_id = 0
        
        for doc in documents:
            text = doc.get('text', '') or ''
            page_num = doc.get('page_num', 1)
            doc_name = doc.get('doc_name', 'Unknown')
            source = doc.get('source', 'Unknown')
            
            sections = self._extract_regulatory_structure(text) if self.respect_boundaries else [{'content': text, 'metadata': {}}]
            
            for section_data in sections:
                section_text = section_data['content']
                section_meta = section_data['metadata']
                
                section_chunks = self._smart_chunk_section(section_text)
                
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
                            'subsection': section_meta.get('subsection', '')
                        })
                        chunk_id += 1
        
        logger.info(f"✅ Created {len(all_chunks)} advanced chunks from {len(documents)} pages")
        return all_chunks
    
    def _extract_regulatory_structure(self, text: str) -> List[Dict]:
        sections = []
        
        patterns = [
            (r'Regulation\s+(\d+[A-Z]*)', 'section'),
            (r'Section\s+(\d+[A-Z]*)', 'section'),
            (r'Schedule\s+([A-Z])', 'section'),
            (r'Part\s+([A-Z])', 'section'),
            (r'(\d+\.\d+\.\d+)(?:\s|$)', 'subsection')
        ]
        
        current_section = {'content': '', 'metadata': {}}
        lines = text.split('\n')
        
        for line in lines:
            matched = False
            for pattern, label in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {
                        'content': line + '\n',
                        'metadata': {label: match.group(1)}
                    }
                    matched = True
                    break
            
            if not matched:
                current_section['content'] += line + '\n'
        
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections if sections else [{'content': text, 'metadata': {}}]
    
    def _smart_chunk_section(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return [text] if len(text.split()) >= self.min_chunk_size else []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(para.split())
            if current_tokens + para_tokens > self.target_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_tokens = len(overlap_text.split())
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def calculate_statistics(self, chunks: List[Dict]) -> Dict:
        token_counts = [c['token_count'] for c in chunks]
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_chunk_size': min(token_counts) if token_counts else 0,
            'max_chunk_size': max(token_counts) if token_counts else 0
        }
