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
                 model_name: str = None,
                 ollama_host: str = None):
        """Initialize Ollama-based answer generator with self-reflection capabilities."""
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "mistral")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.api_endpoint = f"{self.ollama_host}/api/generate"
        
        # Test connectivity
        try:
            response = requests.post(
                f"{self.ollama_host}/api/tags",
                timeout=5
            )
            logger.info(f"✅ Initialized Ollama with model: {self.model_name}")
        except Exception as e:
            raise ValueError(f"Cannot connect to Ollama at {self.ollama_host}. Make sure Ollama is running. Error: {e}")
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_context: Dict,
                       max_tokens: int = 400) -> Dict:
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
        
        context_str = self._build_context_string(chunks[:3])
        
        system_prompt = """You are a compliance Q&A assistant for SEBI documents.

STRICT GROUNDING RULES:
1. ONLY use information EXPLICITLY stated in the provided documents.
2. DO NOT use external knowledge, general facts, or assumptions.
3. If the answer is not in the documents, respond EXACTLY: "Not in documents"
4. QUOTE or closely paraphrase the documents - do not infer or extrapolate.
5. Every factual sentence MUST end with a citation: [Source: doc_name, Page X, Chunk Y]
6. Use doc_name exactly as provided (no .pdf extension).
7. Keep answers concise (2-3 sentences maximum).

FORBIDDEN:
- Making up facts or using general knowledge
- Saying "I'm sorry, I am an AI language model..."
- Providing advice not explicitly in documents"""
        
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
                timeout=300
            )
            response.raise_for_status()
        except requests.exceptions.ReadTimeout:
            logger.warning("Generation timed out. Retrying with smaller context and output length.")
            retry_context = self._build_context_string(chunks[:1])
            retry_prompt = f"{system_prompt}\n\nQuestion: {query}\n\nDocuments:\n{retry_context}\n\nAnswer using only the documents with citations."
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": retry_prompt,
                    "stream": False,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": min(200, max_tokens)
                },
                timeout=180
            )
            response.raise_for_status()
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
        
        result = response.json()
        answer_text = result.get('response', '')
        
        # HALLUCINATION GUARDRAIL: Verify answer is grounded
        if not self._verify_answer_grounded(answer_text, chunks):
            logger.warning("Hallucination detected - answer not grounded in chunks")
            answer_text = "Not in documents (generated answer appears to contain information not present in retrieved context)"
            citations = []
        else:
            citations = self._extract_citations(answer_text, chunks)
        
        if not citations and chunks:
            fallback_chunks = chunks[:2]
            citations = [
                {
                    'doc_name': c['doc_name'],
                    'page': int(c['page']),
                    'chunk_id': str(c['chunk_id'])
                }
                for c in fallback_chunks
            ]
            fallback_sources = " ".join(
                f"[Source: {c['doc_name']}, Page {c['page']}, Chunk {c['chunk_id']}]"
                for c in citations
            )
            answer_text = f"{answer_text}\n\nSources: {fallback_sources}"
        
        return {
            'answer': answer_text,
            'citations': citations,
            'confidence': retrieval_confidence,
            'retrieved_chunks': chunks,
            'model': self.model_name,
            'status': 'success'
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
        patterns = [
            r'\[Source:\s*([^,]+),\s*Page\s*(\d+)(?:-\d+)?,\s*Chunk\s*(\w+)\]',
            r'\[([^,\]]+),\s*Page\s*(\d+)(?:-\d+)?,\s*Chunk\s*(\w+)\]',
            r'\[([^,\]]+\.pdf),\s*Page\s*(\d+)(?:-\d+)?,\s*Chunk\s*(\w+)\]'
        ]
        matches = []
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            if matches:
                break
        
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
    
    def _verify_answer_grounded(self, answer: str, chunks: List[Dict]) -> bool:
        """Verify answer is grounded in retrieved chunks using keyword overlap."""
        
        if not answer or len(answer) < 10:
            return True
        
        # Common hallucination patterns
        hallucination_patterns = [
            "i'm sorry",
            "i am an ai",
            "language model",
            "i don't have access",
            "general knowledge",
            "typically",
            "usually",
            "in general",
            "commonly"
        ]
        
        answer_lower = answer.lower()
        
        # Check for hallucination patterns
        for pattern in hallucination_patterns:
            if pattern in answer_lower:
                return False
        
        # Extract content words from answer (ignore citations)
        answer_clean = re.sub(r'\[Source:.*?\]', '', answer)
        answer_words = set(re.findall(r'\b\w{4,}\b', answer_clean.lower()))
        
        # Extract words from chunks
        chunk_words = set()
        for chunk in chunks:
            words = re.findall(r'\b\w{4,}\b', chunk['text'].lower())
            chunk_words.update(words)
        
        # Calculate overlap
        if not answer_words:
            return True
        
        overlap = len(answer_words & chunk_words) / len(answer_words)
        
        # Require at least 40% keyword overlap
        return overlap >= 0.4
    
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
