# src/indexing/loader.py

import pdfplumber
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, docs_path: str = "./docs"):
        self.docs_path = Path(docs_path)
    
    def load_all_pdfs(self) -> List[Dict]:
        """Load all PDFs from docs folder."""
        documents = []
        
        pdf_files = list(self.docs_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDFs found in {self.docs_path}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF(s)")
        
        for pdf_path in pdf_files:
            logger.info(f"Loading {pdf_path.name}...")
            doc = self.load_single_pdf(pdf_path)
            documents.extend(doc)
        
        return documents
    
    def load_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """Load single PDF and extract text + metadata."""
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"  Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    tables = page.extract_tables()
                    
                    chunks.append({
                        'text': text,
                        'tables': tables,
                        'doc_name': pdf_path.stem,
                        'page_num': page_num + 1,
                        'source': str(pdf_path),
                        'total_pages': total_pages
                    })
        
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")
        
        return chunks

def load_documents(docs_path: str = "./docs") -> List[Dict]:
    """Convenience function to load documents."""
    loader = DocumentLoader(docs_path)
    return loader.load_all_pdfs()