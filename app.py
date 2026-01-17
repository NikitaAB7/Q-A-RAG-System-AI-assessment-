# app.py (Updated for Gemini)

import json
from src.indexing.embeddings import EmbeddingManager
from src.indexing.vector_db import VectorDB
from src.retrieval.retriever import ComplianceRetriever
from src.generation.answer_generator import GroundedAnswerGenerator

def main():
    print("🚀 Initializing Compliance Q&A System with Gemini...\n")
    
    # Initialize components
    embeddings = EmbeddingManager()
    vector_db = VectorDB()
    vector_db.create_collection()
    
    retriever = ComplianceRetriever(vector_db, embeddings)
    
    # Initialize Gemini
    try:
        generator = GroundedAnswerGenerator(model_name="gemini-1.5-flash")
        connectivity = generator.test_connectivity()
        print(f"✅ {connectivity['message']}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("="*80)
    print("COMPLIANCE Q&A SYSTEM (Powered by Google Gemini)")
    print("="*80)
    print("\nAsk your SEBI compliance questions")
    print("(Type 'exit' to quit, 'eval' to run evaluation)\n")
    
    results = []
    
    while True:
        query = input("Q: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'eval':
            run_evaluation(retriever, generator, results)
            continue
        
        if not query:
            continue
        
        print("\n⏳ Processing...\n")
        
        # Retrieve context
        retrieved = retriever.retrieve_with_confidence(query, top_k=5)
        
        # Generate answer
        result = generator.generate_answer(query, retrieved)
        
        # Store for later evaluation
        results.append({
            'question': query,
            'answer': result['answer'],
            'citations': result['citations'],
            'retrieved_chunks': result['retrieved_chunks'],
            'model': result['model'],
            'status': result['status']
        })
        
        # Display results
        print(f"📌 Confidence: {result['confidence'].upper()}")
        print(f"📊 Status: {result['status']}\n")
        print("─" * 80)
        print(f"ANSWER:\n{result['answer']}\n")
        print("─" * 80)
        
        if result['citations']:
            print("\n📚 CITATIONS:")
            for i, cite in enumerate(result['citations'], 1):
                print(f"   {i}. {cite['doc_name']}, Page {cite['page']}, Chunk {cite['chunk_id']}")
        else:
            print("\n📚 No citations (answer based on context but not explicitly sourced)")
        
        print("\n" + "="*80 + "\n")
    
    # Save results
    if results:
        with open('answers.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Saved {len(results)} Q&A results to answers.json")

def run_evaluation(retriever, generator, results):
    """Run evaluation on collected results."""
    print("\n📊 RUNNING EVALUATION...\n")
    
    # Load test questions
    with open('questions.json') as f:
        test_questions = json.load(f)
    
    for q in test_questions[:5]:  # Test first 5
        retrieved = retriever.retrieve_with_confidence(q['question'], top_k=5)
        result = generator.generate_answer(q['question'], retrieved)
        
        print(f"Q: {q['question']}")
        print(f"Status: {result['status']}")
        print(f"Citations: {len(result['citations'])}")
        print("---")

if __name__ == "__main__":
    main()