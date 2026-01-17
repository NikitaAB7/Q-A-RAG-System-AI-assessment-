# src/evaluation/evaluator.py

import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, questions_file: str = "questions.json"):
        with open(questions_file) as f:
            self.questions = json.load(f)
    
    def evaluate_retrieval_recall(self, 
                                 results: List[Dict],
                                 manually_annotated: Dict) -> float:
        """Calculate Recall@5."""
        recalls = []
        
        for i, result in enumerate(results):
            retrieved_ids = {c['chunk_id'] for c in result.get('retrieved_chunks', [])}
            relevant_ids = set(manually_annotated.get(f"q{i}", []))
            
            if not relevant_ids:
                recalls.append(1.0)
                continue
            
            intersection = len(retrieved_ids & relevant_ids)
            recall = intersection / len(relevant_ids)
            recalls.append(recall)
        
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def evaluate_citation_correctness(self, results: List[Dict]) -> Dict:
        """Check if citations match retrieved chunks."""
        correct = 0
        total = 0
        
        for result in results:
            for cite in result.get('citations', []):
                found = any(
                    c['doc_name'] == cite['doc_name'] and c['page'] == cite['page']
                    for c in result.get('retrieved_chunks', [])
                )
                correct += found
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            'correct_citations': correct,
            'total_citations': total,
            'citation_accuracy': accuracy
        }
    
    def generate_report(self, results: List[Dict], 
                       output_file: str = "output/evaluation_report.json"):
        """Generate comprehensive evaluation report."""
        citation_eval = self.evaluate_citation_correctness(results)
        
        report = {
            'total_questions': len(results),
            'successful_answers': sum(1 for r in results if r.get('status') == 'success'),
            'citations_metrics': citation_eval,
            'sample_results': results[:5]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Evaluation report saved to {output_file}")
        return report