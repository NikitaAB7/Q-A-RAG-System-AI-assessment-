import streamlit as st
from typing import Dict
from src.indexing.embeddings import EmbeddingManager
from src.retrieval.vector_db import VectorDB
from src.retrieval.retriever import ComplianceRetriever
from src.generation.answer_generator import GroundedAnswerGenerator


st.set_page_config(
    page_title="SEBI Compliance QA",
    page_icon="📘",
    layout="wide"
)


@st.cache_resource(show_spinner=True)
def load_components():
    embeddings = EmbeddingManager()
    vector_db = VectorDB(db_path="./chromadb")
    vector_db.create_collection("compliance_docs")
    retriever = ComplianceRetriever(vector_db, embeddings)
    generator = GroundedAnswerGenerator()
    return retriever, generator


def render_citations(citations):
    if not citations:
        st.info("No explicit citations detected. The answer may be grounded but did not include citation tags.")
        return

    for i, cite in enumerate(citations, 1):
        st.write(f"{i}. {cite['doc_name']} — Page {cite['page']}, Chunk {cite['chunk_id']}")


def render_chunks(chunks):
    if not chunks:
        st.warning("No chunks retrieved. Try a different query or re-index documents.")
        return

    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('rerank_score', chunk.get('hybrid_score', chunk.get('score', 0)))
        section = chunk.get('section', '')
        subsection = chunk.get('subsection', '')
        location = section or 'N/A'
        if subsection:
            location = f"{location} → {subsection}"

        with st.expander(f"Chunk {i}: {chunk['doc_name']} (Page {chunk['page']}) — Score {score:.3f}"):
            st.write(f"Location: {location}")
            st.write(chunk['text'])


def main():
    st.title("📘 SEBI Compliance RAG Assistant")
    st.write("Ask compliance questions and get grounded answers with citations.")

    with st.sidebar:
        st.header("Query Settings")
        top_k = st.slider("Top K", min_value=3, max_value=10, value=5)
        rerank_top_k = st.slider("Rerank Candidates", min_value=10, max_value=100, value=50)
        dense_weight = st.slider("Dense Weight", 0.0, 1.0, 0.6, 0.05)
        sparse_weight = st.slider("Sparse Weight", 0.0, 1.0, 0.4, 0.05)
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.slider("Max Answer Tokens", 100, 600, 300, 50)
        doc_filter = st.text_input("Filter by Document (optional)")
        clear_history = st.button("Clear History")

    if clear_history:
        st.session_state.pop("history", None)

    query = st.text_area("Your question", placeholder="Ask a SEBI compliance question...")
    ask = st.button("Ask")

    if ask and query.strip():
        with st.spinner("Retrieving context and generating answer..."):
            retriever, generator = load_components()

            metadata_filter = None
            if doc_filter.strip():
                metadata_filter = {"doc_name": doc_filter.strip()}

            retrieved = retriever.retrieve_with_reranking(
                query,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                score_threshold=score_threshold,
                rerank_top_k=rerank_top_k,
                metadata_filter=metadata_filter
            )

            result = generator.generate_answer(query, retrieved, max_tokens=max_tokens)

        st.subheader("Answer")
        st.write(result['answer'])

        st.subheader("Status")
        st.write(f"Confidence: **{result['confidence']}**")
        st.write(f"Status: **{result['status']}**")
        st.write(f"Model: **{result['model']}**")

        st.subheader("Citations")
        render_citations(result['citations'])

        st.subheader("Retrieved Chunks")
        render_chunks(result['retrieved_chunks'])

        history = st.session_state.get("history", [])
        history.append({
            "question": query,
            "answer": result['answer'],
            "citations": result['citations'],
            "confidence": result['confidence'],
            "status": result['status']
        })
        st.session_state["history"] = history

    if st.session_state.get("history"):
        st.subheader("History")
        for i, item in enumerate(reversed(st.session_state["history"]), 1):
            with st.expander(f"Q{i}: {item['question']}"):
                st.write(item['answer'])
                st.write(f"Confidence: {item['confidence']} | Status: {item['status']}")
                render_citations(item['citations'])


if __name__ == "__main__":
    main()