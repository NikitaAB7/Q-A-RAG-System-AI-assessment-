# Run the UI (Streamlit)

This guide explains how to run the SEBI Compliance RAG UI on any PC.

---

## ✅ Prerequisites
- Python 3.10+ installed
- Ollama installed and running
- Documents placed in the docs/ folder

---

## 1) Setup

Create and activate a virtual environment, then install dependencies:

### Windows (PowerShell)
```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Pull an LLM Model (Ollama)

```
ollama pull phi
```

---

## 3) Index Documents (First-time only)

```
python index.py
```

---

## 4) Launch the UI

```
streamlit run app.py
```

The UI opens in your browser at:

```
http://localhost:8501
```

---

## Optional Configuration

Copy .env.example to .env and edit settings:

```
OLLAMA_MODEL=phi
OLLAMA_HOST=http://localhost:11434
```

---

## Troubleshooting

**Ollama not responding**
- Ensure Ollama is running: `ollama serve`

**No results or empty answers**
- Re-run `python index.py`
- Confirm PDFs exist in docs/

**Streamlit command not found**
- Ensure venv is activated
- Reinstall requirements: `pip install -r requirements.txt`

---

## Notes
- First run after indexing may take longer
- Use smaller model (phi) for speed, larger model (mistral) for quality
