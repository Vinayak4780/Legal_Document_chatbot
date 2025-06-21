import ragas
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.evaluation import evaluate

# Example usage for RAG evaluation
# Provide your list of queries, answers, and retrieved contexts
queries = [
    "What is the privacy policy regarding user data?",
    "How can a user terminate their contract?"
]
answers = [
    "The privacy policy states that user data will not be shared without consent.",
    "A user can terminate their contract by providing a 30-day written notice."
]
contexts = [
    ["User data is kept confidential and not shared without explicit consent."],
    ["To terminate the contract, a 30-day written notice is required."]
]

# Evaluate using RAGAS metrics
results = evaluate(
    queries=queries,
    answers=answers,
    contexts=contexts,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print("RAGAS Evaluation Results:")
print(results)
