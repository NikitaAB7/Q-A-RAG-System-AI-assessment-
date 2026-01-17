# src/generation/answer_generator.py

import os
import re
import requests
import json
from typing import List, Dict
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class GroundedAnswerGenerator:
    def __init__(self, api_key: str = None, 
                 model_name: str = "mistral",
                 ollama_host: str = None):
        """Initialize Ollama-based answer generator with Mistral."""
        self.model_name = model_name
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.api_endpoint = f"{self.ollama_host}/api/generate"
        
        # Test connectivity
        try:
            response = requests.post(
                f"{self.ollama_host}/api/tags",
                timeout=5
            )
            logger.info(f"✅ Initialized Ollama with model: {model_name}")
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama at {self.ollama_host}. Make sure Ollama is running. Error: {e}")
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_context: Dict,
                       max_tokens: int = 1500) -> Dict:
        """Generate grounded answer using Mistral via Ollama."""
        
        chunks = retrieved_context['chunks']
        retrieval_confidence = retrieved_context['confidence']
        
        if not chunks or retrieval_confidence == 'low':
            return {
                'answer': "Information not found in provided documents.",
                'citations': [],
                'confidence': 'not_found',
                'retrieved_chunks': chunks,
                'model': self.model_name,
                'status': 'low_confidence'
            }
        
        context_str = self._build_context_string(chunks)
        
        system_prompt = """You are a compliance Q&A assistant for SEBI documents.

RULES:
1. Answer ONLY based on provided documents.
2. If not in docs, say: "Not in documents"
3. Cite sources as [Source: doc_name, Page X, Chunk Y]
4. Be concise and precise."""
        
        user_message = f"""Question: {query}

Documents:
{context_str}

Answer using only the documents with citations."""
        
        full_prompt = f"{system_prompt}\n\n{user_message}"
        
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": max_tokens
                },
                timeout=120  # Increased timeout for longer responses
            )
            response.raise_for_status()
            
            result = response.json()
            answer_text = result.get('response', '')
            citations = self._extract_citations(answer_text, chunks)
            
            return {
                'answer': answer_text,
                'citations': citations,
                'confidence': retrieval_confidence,
                'retrieved_chunks': chunks,
                'model': self.model_name,
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'citations': [],
                'confidence': 'error',
                'retrieved_chunks': chunks,
                'model': self.model_name,
                'status': 'error'
            }
    
    def _build_context_string(self, chunks: List[Dict]) -> str:
        """Build context string with attribution."""
        context_parts = []
        
        for chunk in chunks:
            header = f"[{chunk['doc_name']}, Page {chunk['page']}, Chunk {chunk['chunk_id']}, Score: {chunk['score']}]"
            context_parts.append(f"{header}\n{chunk['text']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """Extract citations from answer - supports multiple formats."""
        # Try pattern 1: [Source: doc_name, Page X, Chunk Y]
        pattern1 = r'\[Source:\s*([^,]+),\s*Page\s*(\d+),\s*Chunk\s*(\w+)\]'
        matches = re.findall(pattern1, answer)
        
        # Try pattern 2: [doc_name, Page X, Chunk Y] (what Mistral actually uses)
        if not matches:
            pattern2 = r'\[([^,\]]+\.pdf),\s*Page\s*(\d+),\s*Chunk\s*(\w+)\]'
            matches = re.findall(pattern2, answer)
        
        citations = []
        seen = set()
        
        for doc_name, page, chunk_id in matches:
            key = (doc_name.strip(), int(page), chunk_id.strip())
            if key not in seen:
                citations.append({
                    'doc_name': doc_name.strip(),
                    'page': int(page),
                    'chunk_id': chunk_id.strip()
                })
                seen.add(key)
        
        return citations
    
    def test_connectivity(self) -> Dict:
        """Test Ollama connectivity."""
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return {
                'status': 'success',
                'model': self.model_name,
                'message': 'Ollama working correctly'
            }
        except Exception as e:
            return {
                'status': 'error',
                'model': self.model_name,
                'error': str(e)
            }