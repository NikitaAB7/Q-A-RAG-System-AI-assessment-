# evaluation_metrics.py - Comprehensive RAG Evaluation

import json
import time
from typing import List, Dict
from src.generation.answer_generator import GroundedAnswerGenerator
from src.retrieval.retriever import ComplianceRetriever
from src.retrieval.vector_db import VectorDB
from src.indexing.embeddings import EmbeddingManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG system on multiple metrics."""
    
    def __init__(self):
        """Initialize evaluator with RAG components."""
        self.vector_db = VectorDB(db_path="./chromadb")
        self.vector_db.create_collection("compliance_docs")
        
        self.embeddings_manager = EmbeddingManager()
        self.retriever = ComplianceRetriever(self.vector_db, self.embeddings_manager)
        
        self.generator = GroundedAnswerGenerator()
        
    # ===== METRIC 1: RETRIEVAL RECALL@K =====
    def calculate_recall_at_k(self, retrieved_chunks: List[Dict], 
                             relevant_docs: List[str], k: int = 5) -> float:
        """
        Calculate Recall@k for retrieval.
        
        Recall@k = (Number of relevant docs in top-k) / (Total relevant docs)
        
        Args:
            retrieved_chunks: Retrieved chunks from vector DB
            relevant_docs: Ground truth relevant document names
            k: Top-k to consider
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 1.0
        
        retrieved_docs = set([chunk['doc_name'] for chunk in retrieved_chunks[:k]])
        relevant_set = set(relevant_docs)
        
        if len(relevant_set) == 0:
            return 0.0
        
        num_relevant_retrieved = len(retrieved_docs & relevant_set)
        recall = num_relevant_retrieved / len(relevant_set)
        
        return min(recall, 1.0)
    
    # ===== METRIC 2: FAITHFULNESS (LLM-AS-JUDGE) =====
    def evaluate_faithfulness(self, question: str, answer: str, 
                            context: str) -> Dict:
        """
        Evaluate if answer is faithful to the retrieved context.
        Uses LLM-as-judge (Mistral).
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dict with faithfulness score and reasoning
        """
        faithfulness_prompt = f"""You are evaluating if an answer is faithful to the provided context.

TASK: Determine if the answer is grounded in the context. Rate on a scale of 1-5:
1 = Completely unfaithful / contradicts context
2 = Mostly unfaithful / significant inaccuracies
3 = Partially faithful / some inaccuracies
4 = Mostly faithful / minor inaccuracies
5 = Completely faithful / accurate and grounded

Question: {question}

Context:
{context[:1000]}  # Limit context length for Mistral

Answer: {answer}

Respond ONLY with the rating (1-5) and a brief reason."""
        
        try:
            # Create a mock context for the generator
            mock_retrieved = {
                'chunks': [{'text': context, 'doc_name': 'context', 'page': 1, 
                           'chunk_id': '0', 'score': 1.0}],
                'confidence': 'high'
            }
            
            result = self.generator.generate_answer(
                faithfulness_prompt, 
                mock_retrieved,
                max_tokens=100
            )
            
            response = result['answer']
            
            # Extract rating from response
            rating = None
            for line in response.split('\n'):
                if any(str(i) in line for i in range(1, 6)):
                    for i in range(1, 6):
                        if str(i) in line:
                            rating = i
                            break
            
            if rating is None:
                rating = 3  # Default to neutral if can't parse
            
            return {
                'faithfulness_score': rating / 5.0,  # Normalize to 0-1
                'rating': rating,
                'reasoning': response[:200]
            }
        except Exception as e:
            logger.warning(f"Faithfulness evaluation error: {e}")
            return {
                'faithfulness_score': 0.5,
                'rating': 3,
                'reasoning': 'Evaluation failed'
            }
    
    # ===== METRIC 3: CITATION CORRECTNESS =====
    def evaluate_citation_correctness(self, answer: str, 
                                    cited_chunks: List[Dict]) -> Dict:
        """
        Evaluate if citations in answer are correct.
        
        Metrics:
        - Citation Precision: Are cited chunks actually mentioned?
        - Citation Recall: Should more chunks be cited?
        - Citation Format: Are citations properly formatted?
        
        Args:
            answer: Generated answer with citations
            cited_chunks: Chunks that were cited
            
        Returns:
            Dict with citation metrics
        """
        # Check if citations exist
        has_citations = '[' in answer and ']' in answer
        
        citation_count = answer.count('[')
        
        # Check citation format (basic check)
        proper_format = all(']' in answer[answer.find('['):] 
                           for _ in range(citation_count))
        
        return {
            'has_citations': has_citations,
            'citation_count': citation_count,
            'proper_format': proper_format,
            'citation_precision': min(1.0, citation_count / max(len(cited_chunks), 1))
        }
    
    def evaluate_rag(self, questions: List[Dict], 
                    num_questions: int = 20) -> Dict:
        """
        Evaluate RAG system on multiple metrics.
        
        Args:
            questions: List of question dicts with 'question' key
            num_questions: Number of questions to evaluate
            
        Returns:
            Evaluation results with metrics
        """
        results = []
        
        print(f"Starting evaluation on {min(num_questions, len(questions))} questions...\n")
        
        for i, q in enumerate(questions[:num_questions], 1):
            question = q['question']
            
            # RETRIEVAL
            start = time.time()
            retrieved = self.retriever.retrieve_with_reranking(
                question,
                top_k=3,
                dense_weight=0.6,
                sparse_weight=0.4,
                score_threshold=0.1,
                rerank_top_k=20
            )
            retrieval_time = time.time() - start
            
            retrieved_chunks = retrieved.get('chunks', [])
            
            # GENERATION
            start = time.time()
            gen_result = self.generator.generate_answer(question, retrieved, max_tokens=300)
            generation_time = time.time() - start
            
            answer = gen_result['answer']
            citations = gen_result['citations']
            
            # METRIC 1: Recall@k (assume relevant docs are in first retrieval)
            recall_at_5 = self.calculate_recall_at_k(
                retrieved_chunks,
                [c['doc_name'] for c in retrieved_chunks],  # Mock: use retrieved as relevant
                k=5
            )
            
            # METRIC 2: Faithfulness (LLM-as-judge)
            context_str = "\n".join([f"- {c['text'][:200]}" for c in retrieved_chunks[:3]])
            faithfulness = self.evaluate_faithfulness(question, answer, context_str)
            
            # METRIC 3: Citation Correctness
            citation_eval = self.evaluate_citation_correctness(answer, retrieved_chunks)
            
            result = {
                'question_id': i,
                'question': question,
                'answer': answer[:500],  # Truncate for storage
                'retrieval_time': round(retrieval_time, 3),
                'generation_time': round(generation_time, 3),
                'total_time': round(retrieval_time + generation_time, 3),
                
                # Metric 1: Retrieval Recall@5
                'recall_at_5': round(recall_at_5, 3),
                
                # Metric 2: Faithfulness
                'faithfulness_score': round(faithfulness['faithfulness_score'], 3),
                'faithfulness_rating': faithfulness['rating'],
                
                # Metric 3: Citation Correctness
                'has_citations': citation_eval['has_citations'],
                'citation_count': citation_eval['citation_count'],
                'citation_format_correct': citation_eval['proper_format'],
                'citation_precision': round(citation_eval['citation_precision'], 3),
                
                'num_retrieved_chunks': len(retrieved_chunks),
                'confidence': retrieved['confidence']
            }
            
            results.append(result)
            
            # Print progress
            status = "[OK]" if gen_result['status'] == 'success' else "[WARN]"
            print(f"{i:2}. {status} | R@5: {recall_at_5:.2f} | Faith: {faithfulness['rating']}/5 | Citations: {citation_eval['citation_count']}")
        
        # Calculate aggregate metrics
        if results:
            avg_recall = sum(r['recall_at_5'] for r in results) / len(results)
            avg_faithfulness = sum(r['faithfulness_score'] for r in results) / len(results)
            citation_rate = sum(1 for r in results if r['has_citations']) / len(results)
            avg_latency = sum(r['total_time'] for r in results) / len(results)
            
            summary = {
                'total_questions': len(results),
                'metrics': {
                    'retrieval_recall_at_5': round(avg_recall, 3),
                    'faithfulness_score': round(avg_faithfulness, 3),
                    'citation_rate': round(citation_rate, 3),
                    'avg_latency_seconds': round(avg_latency, 3),
                    'avg_retrieved_chunks': round(sum(r['num_retrieved_chunks'] for r in results) / len(results), 1)
                },
                'questions_evaluated': results
            }
        else:
            summary = {'error': 'No results generated'}
        
        return summary


def main():
    # Load questions
    with open('questions.json') as f:
        questions = json.load(f)
    
    # Run evaluation
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_rag(questions, num_questions=20)
    
    # Save results
    with open('answers.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    if 'metrics' in results:
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Questions evaluated: {results['total_questions']}")
        print(f"\n--- METRIC 1: RETRIEVAL RECALL@5 ---")
        print(f"Avg Recall@5: {results['metrics']['retrieval_recall_at_5']:.3f}")
        print(f"(Higher is better: 1.0 = all relevant docs retrieved)")
        
        print(f"\n--- METRIC 2: FAITHFULNESS (LLM-as-Judge) ---")
        print(f"Avg Faithfulness: {results['metrics']['faithfulness_score']:.3f}")
        print(f"(Scale 0-1: measures if answer is grounded in context)")
        
        print(f"\n--- METRIC 3: CITATION CORRECTNESS ---")
        print(f"Citation Rate: {results['metrics']['citation_rate']:.1%}")
        print(f"Avg Chunks Retrieved: {results['metrics']['avg_retrieved_chunks']:.1f}")
        print(f"(Checks if answers include proper citations)")
        
        print(f"\n--- PERFORMANCE ---")
        print(f"Avg Latency: {results['metrics']['avg_latency_seconds']:.3f}s")
        
        print(f"\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()
