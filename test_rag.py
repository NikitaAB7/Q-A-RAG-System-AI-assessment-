# test_rag.py

from src.generation.answer_generator import GroundedAnswerGenerator

# Initialize generator with Mistral via Ollama
generator = GroundedAnswerGenerator(model_name="mistral")

query = "What is KYC according to SEBI regulations?"
retrieved = {
    'chunks': [
        {
            'text': 'KYC stands for Know Your Customer. It is mandatory for all investment intermediaries.',
            'doc_name': 'sebi_circular_1.pdf',
            'page': 15,
            'chunk_id': '42',
            'score': 0.92
        }
    ],
    'confidence': 'high'
}

result = generator.generate_answer(query, retrieved)
print(result['answer'])
print(f"Citations: {result['citations']}")