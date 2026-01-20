# query.py - Query the RAG system

from src.generation.answer_generator import GroundedAnswerGenerator
from src.retrieval.retriever import ComplianceRetriever
from src.retrieval.vector_db import VectorDB
from src.indexing.embeddings import EmbeddingManager
import logging

logging.basicConfig(level=logging.INFO)

def query_rag(question: str):
    """Query the RAG system with a question."""
    
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print("="*80)
    
    # Initialize components
    print("\n[1/3] Loading RAG components...")
    vector_db = VectorDB(db_path="./chromadb")
    vector_db.create_collection("compliance_docs")
    
    embeddings_manager = EmbeddingManager()
    retriever = ComplianceRetriever(vector_db, embeddings_manager)
    generator = GroundedAnswerGenerator()
    
    # Retrieve relevant chunks
    print("\n[2/3] Retrieving relevant context...")
    retrieved = retriever.retrieve_with_reranking(
        question,
        top_k=5,
        dense_weight=0.6,
        sparse_weight=0.4,
        score_threshold=0.1,
        rerank_top_k=50
    )
    
    print(f"Retrieved {len(retrieved['chunks'])} chunks")
    print(f"Confidence: {retrieved['confidence']}")
    
    if retrieved['chunks']:
        print("\nTop 3 chunks:")
        for i, chunk in enumerate(retrieved['chunks'][:3], 1):
            section = chunk.get('section', '')
            subsection = chunk.get('subsection', '')
            location = section or 'N/A'
            if subsection:
                location = f"{location} → {subsection}"
            score = chunk.get('rerank_score', chunk.get('hybrid_score', chunk.get('score', 0)))
            print(f"  {i}. [{chunk['doc_name']}, Page {chunk['page']}, {location}] Score: {score:.3f}")
            print(f"     {chunk['text'][:150]}...")
    
    # Generate answer
    print("\n[3/3] Generating answer...")
    result = generator.generate_answer(question, retrieved, max_tokens=300)
    
    # Display results
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(result['answer'])
    
    if result['citations']:
        print(f"\n--- CITATIONS ({len(result['citations'])}) ---")
        for i, cite in enumerate(result['citations'], 1):
            print(f"{i}. {cite['doc_name']}, Page {cite['page']}, Chunk {cite['chunk_id']}")
    else:
        print("\n--- NO EXPLICIT CITATIONS FOUND ---")
    
    print("\n" + "="*80)
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Model: {result['model']}")
    print("="*80 + "\n")
    
    return result


if __name__ == "__main__":
    # Example queries
    questions = [
        "What is the maximum exposure limit of a mutual fund scheme to equity derivatives as per SEBI regulations?",
        "Are index options considered part of the derivatives exposure limit or treated separately?",
        "Can a mutual fund scheme write naked call options? If not, what are the constraints?"
    ]
    
    print("RAG SYSTEM - INTERACTIVE QUERY")
    print("="*80)
    print("\nTesting with sample questions...\n")
    
    for i, q in enumerate(questions[:2], 1):  # Test with 2 questions
        print(f"\n### Query {i} ###")
        query_rag(q)
        print("\n" + "-"*80 + "\n")
    
    print("\n\nTo query interactively, modify the questions list in query.py")
    print("Or run: python -c \"from query import query_rag; query_rag('your question here')\"")
