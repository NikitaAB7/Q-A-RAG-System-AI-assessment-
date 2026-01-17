# evaluate.py - Test on 20 questions

import json
import time
from src.generation.answer_generator import GroundedAnswerGenerator
from src.retrieval.retriever import ComplianceRetriever
from src.retrieval.vector_db import VectorDB
from src.indexing.embeddings import EmbeddingManager

def evaluate_rag():
    # Initialize components
    vector_db = VectorDB(db_path="./chromadb")
    vector_db.create_collection("compliance_docs")
    
    embeddings_manager = EmbeddingManager()
    retriever = ComplianceRetriever(vector_db, embeddings_manager)
    
    generator = GroundedAnswerGenerator()
    
    # Load test questions
    with open('questions.json') as f:
        questions = json.load(f)
    
    results = []
    total_tokens = 0
    
    print("Running evaluation on 20 questions...\n")
    
    for i, q in enumerate(questions[:20], 1):
        # Retrieve
        retrieved = retriever.retrieve_with_confidence(q['question'], top_k=5)
        
        # Generate
        start = time.time()
        result = generator.generate_answer(q['question'], retrieved)
        latency = time.time() - start
        
        results.append({
            'question': q['question'],
            'answer': result['answer'],
            'citations': result['citations'],
            'latency': latency,
            'tokens': result.get('output_tokens', 0)
        })
        
        total_tokens += result.get('output_tokens', 0)
        
        status = "[OK]" if result['status'] == 'success' else "[WARN]"
        print(f"{i:2}. {status} {latency:.2f}s - {len(result['citations'])} citations")
    
    # Save results
    with open('answers.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    avg_latency = sum(r['latency'] for r in results) / len(results)
    avg_citations = sum(len(r['citations']) for r in results) / len(results)
    
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Questions: {len(results)}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Avg Citations: {avg_citations:.1f}")
    print(f"Total Tokens (estimate): {total_tokens}")
    print(f"\n=== COST ESTIMATE ===")
    print(f"Cost: ${total_tokens * 0.075 / 1_000_000:.4f} (if using paid tier)")

if __name__ == "__main__":
    evaluate_rag()