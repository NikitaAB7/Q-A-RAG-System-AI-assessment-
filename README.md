# QA-RAG System - SEBI Compliance Q&A

A production-ready **Retrieval-Augmented Generation (RAG)** system for answering SEBI (Securities and Exchange Board of India) compliance questions with proper citations and grounding.

## Features

✅ **Semantic Document Retrieval** - Fast vector similarity search using ChromaDB  
✅ **Grounded Generation** - Answers based only on indexed documents  
✅ **Automatic Citations** - Source documents and page numbers included  
✅ **Local LLM Support** - Runs Mistral/Phi locally via Ollama (no API costs)  
✅ **Comprehensive Evaluation** - 3 metrics (Recall, Faithfulness, Citations)  
✅ **Easy to Extend** - Modular architecture for custom implementations  

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) (for local LLM)
- 8GB+ RAM recommended

### 1. Clone & Setup
```bash
git clone <your-repo>
cd QA-RAG-System

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama & Model
```bash
# Download Ollama from https://ollama.ai
# Then pull a model
ollama pull phi      # Fast (2.7B) - recommended for RAG
# OR
ollama pull mistral  # Better quality (7B)
```

### 3. Index Documents
```bash
python index.py
```
Output:
```
[1/4] Loading PDF documents... ✓
[2/4] Chunking documents... ✓ (479 chunks)
[3/4] Initializing embeddings... ✓
[4/4] Embedding and indexing... ✓
```

### 4. Query the System
```bash
python query.py
```
Or programmatically:
```python
from query import query_rag

result = query_rag("What is the maximum exposure limit for equity derivatives?")
print(result['answer'])
print(result['citations'])
```

### 5. Evaluate
```bash
python evaluation_metrics.py
```

## Architecture

```
PDFs → Loader → Chunker → Embeddings → Vector DB (ChromaDB)
                                            ↓
                                       Retriever
                                            ↓
                    Question → Embedding → Top-K Semantic Search
                                            ↓
                                    Retrieved Context
                                            ↓
                    LLM (Mistral/Phi) → Grounded Answer + Citations
```

## Configuration

### `.env` Configuration
```bash
# LLM Backend
LLM_BACKEND=ollama              # Options: ollama, gemini, openai, groq
OLLAMA_MODEL=phi                # Options: phi, mistral, llama2, llama3

# Embeddings
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DEVICE=cpu            # Options: cpu, cuda, mps

# Retrieval
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.3
```

Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
```

## Project Structure

```
QA-RAG-System/
├── index.py                    # Indexing pipeline
├── query.py                    # Query interface
├── evaluation_metrics.py       # Evaluation framework
├── questions.json              # 20 test questions
├── config.yaml                 # System configuration
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── EVALUATION.md              # Evaluation docs
├── README_SYSTEM.md           # System architecture docs
├── src/
│   ├── indexing/
│   │   ├── loader.py          # PDF loading
│   │   ├── chunking.py        # Text chunking
│   │   └── embeddings.py      # Embeddings model
│   ├── retrieval/
│   │   ├── vector_db.py       # ChromaDB interface
│   │   └── retriever.py       # Semantic search
│   ├── generation/
│   │   └── answer_generator.py # Answer generation
│   └── evaluation/
│       └── evaluator.py       # Evaluation metrics
├── docs/                       # PDF documents to index
│   ├── SEBI Regulation.pdf
│   ├── SEBI Master circular for MF.pdf
│   └── Mirae Asset Small Cap SID.pdf
└── chromadb/                   # Vector database (generated)
```

## Usage Examples

### Basic Query
```python
from query import query_rag

result = query_rag("Are index options covered under derivatives exposure limit?")
```

Output:
```
QUESTION: Are index options covered under derivatives exposure limit?

Retrieved 5 chunks
Confidence: medium

ANSWER:
According to SEBI regulations, index options are treated separately from 
equity derivatives exposure limits [SEBI Master circular, Page 142, Chunk 234].

CITATIONS:
1. SEBI Master circular for MF, Page 142, Chunk 234
```

### Batch Evaluation
```bash
python evaluation_metrics.py
```

Generates `evaluation_results.json` with:
- Retrieval Recall@5
- Faithfulness scores
- Citation accuracy
- Latency metrics

## Model Comparison

| Model | Speed | Quality | Memory | Best Use |
|-------|-------|---------|--------|----------|
| Phi 2.7B | ⚡⚡⚡⚡ | ⭐⭐⭐ | 5GB | **RAG (recommended)** |
| Mistral 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | 14GB | Quality over speed |
| Llama3 8B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 16GB | Best quality |

### Switch Models
```bash
ollama pull phi
# Update .env: OLLAMA_MODEL=phi
python query.py
```

## Evaluation Metrics

### 1. Retrieval Recall@5
- Measures if relevant documents are found in top-5
- Range: 0.0 (none found) to 1.0 (all found)
- Target: > 0.80

### 2. Faithfulness (LLM-as-Judge)
- Uses LLM to rate answer grounding in context
- Scale: 0-1 (normalized from 1-5 rating)
- Target: > 0.70

### 3. Citation Correctness
- % of answers with citations
- Format validation
- Source accuracy
- Target: > 0.90

## Performance

| Metric | Value |
|--------|-------|
| Indexing Time | ~3 min (479 chunks) |
| Query Retrieval | ~0.2s (embedding + search) |
| Answer Generation | ~0.5-2s (Phi 2.7B) |
| Total Latency | ~1-3s per query |
| Storage | ~50MB (embeddings + metadata) |

## Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama list

# Restart Ollama if needed
ollama serve
```

### Low Retrieval Scores
- Question phrasing differs from documents
- Try reducing `RETRIEVAL_SCORE_THRESHOLD`
- Consider query expansion in system prompt

### Generation Timeout
- Switch to faster model: `ollama pull phi`
- Reduce `RETRIEVAL_TOP_K`
- Lower max_tokens in generation config

### GPU Out of Memory
- Set `EMBEDDING_DEVICE=cpu`
- Reduce `RETRIEVAL_TOP_K`
- Use smaller embedding model

## Adding Custom Documents

1. Place PDFs in `./docs/`
2. Run indexing:
```bash
python index.py
```
3. Query with new documents:
```bash
python query.py
```

## API (Optional - FastAPI)

To expose as REST API:
```python
from fastapi import FastAPI
from query import query_rag

app = FastAPI()

@app.post("/query")
async def api_query(question: str):
    return query_rag(question)

# Run: uvicorn app:app --reload
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit pull request

## License

MIT License - See LICENSE file for details

## References

- [SEBI Regulations](https://www.sebi.gov.in/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.ai/)
- [Mistral AI](https://www.mistral.ai/)

## Support

For issues and questions:
- Open GitHub issues
- Check EVALUATION.md for evaluation details
- See README_SYSTEM.md for architecture details

---

**Built for FInsharpe Assessment - January 2026**

System Status: ✅ Production Ready
