from src.generation.answer_generator import GroundedAnswerGenerator

generator = GroundedAnswerGenerator()
status = generator.test_connectivity()
print(status)