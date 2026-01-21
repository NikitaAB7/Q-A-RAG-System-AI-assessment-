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
        """Load single PDF and extract text + metadata with table handling."""
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"  Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with layout preservation
                    text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2)
                    
                    if not text:
                        text = ""
                    
                    # Extract tables and convert to markdown format
                    tables = page.extract_tables()
                    table_text = self._tables_to_text(tables)
                    
                    # Combine text and tables
                    full_text = self._clean_text(text)
                    if table_text:
                        full_text += "\n\n" + table_text
                    
                    chunks.append({
                        'text': full_text,
                        'doc_name': pdf_path.stem,
                        'page_num': page_num + 1,
                        'source': str(pdf_path),
                        'total_pages': total_pages,
                        'has_tables': len(tables) > 0
                    })
        
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")
        
        return chunks
    
    def _tables_to_text(self, tables: List) -> str:
        """Convert extracted tables to readable text format."""
        if not tables:
            return ""
        
        table_texts = []
        for i, table in enumerate(tables, 1):
            if not table:
                continue
            
            # Convert table to markdown-like format
            table_str = f"\n[TABLE {i}]\n"
            for row in table:
                if row:
                    # Clean and join cells
                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                    table_str += " | ".join(clean_row) + "\n"
            
            table_texts.append(table_str)
        
        return "\n".join(table_texts)
    
    def _clean_text(self, text: str) -> str:
        """Clean text: fix broken lines, remove headers/footers."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip common headers/footers (page numbers, repeated headers)
            if self._is_header_footer(line, i, len(lines)):
                continue
            
            # Fix broken lines: merge if line doesn't end with punctuation
            if cleaned_lines and line and not line[0].isupper():
                # Likely continuation of previous line
                if cleaned_lines[-1] and cleaned_lines[-1][-1] not in '.!?;:':
                    cleaned_lines[-1] += " " + line
                    continue
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
    
    def _is_header_footer(self, line: str, line_num: int, total_lines: int) -> bool:
        """Detect if line is likely a header/footer."""
        # Page numbers (simple pattern)
        if line.isdigit() and len(line) <= 3:
            return True
        
        # Very short lines at top/bottom of page
        if (line_num < 2 or line_num > total_lines - 3) and len(line) < 30:
            # Common header/footer patterns
            if any(pattern in line.lower() for pattern in ['page', 'confidential', '©', 'all rights reserved']):
                return True
        
        return False

def load_documents(docs_path: str = "./docs") -> List[Dict]:
    """Convenience function to load documents."""
    loader = DocumentLoader(docs_path)
    return loader.load_all_pdfs()