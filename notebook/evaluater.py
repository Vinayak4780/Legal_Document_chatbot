import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrival import response_with_sources
import time
from difflib import SequenceMatcher

def evaluate_rag_system():
  
    # Test questions with realistic expected answers from eBay User Agreement
    test_cases = [
        {
            "question": "How is my personal information governed?",
            "expected_answer": "Your personal information is governed by the User Privacy Notice. The collection, use, disclosure, retention, and protection of your personal information is governed by our User Privacy Notice."
        },
        {
            "question": "How do I opt out of the Agreement to Arbitrate?",
            "expected_answer": "To opt out of the Agreement to Arbitrate, you must postmark a written opt-out notice to eBay Inc., ATTN: Litigation Department, RE: OPT-OUT NOTICE, 583 West eBay Way, Draper, UT 84020. The opt-out notice must be postmarked no later than 30 days from the date you first accept this User Agreement."
        },
        {
            "question": "Can eBay contact me for marketing purposes?",
            "expected_answer": "Yes, eBay may contact you using autodialed or prerecorded calls and text messages for marketing purposes such as offers and promotions, with your consent to such communication."
        }
    ]
    
    print("🏛️  Legal Document RAG System Evaluation")
    print("=" * 50)
    
    total_score = 0
    detailed_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}")
        print(f"❓ Question: {test_case['question']}")
        
        # Measure response time
        start_time = time.time()
        
        try:
            # Get response from RAG system
            result = response_with_sources(test_case['question'])
            generated_answer = result['answer']
            sources = result['sources']
            
            response_time = time.time() - start_time
            
            # Calculate similarity between generated and expected answer
            similarity_score = calculate_similarity(generated_answer, test_case['expected_answer'])
            total_score += similarity_score
            
            print(f"🤖 Generated Answer: {generated_answer[:150]}...")
            print(f"✅ Expected Answer: {test_case['expected_answer'][:150]}...")
            print(f"📊 Similarity Score: {similarity_score:.2f}/10")
            print(f"⏱️  Response Time: {response_time:.2f}s")
            print(f"📄 Sources Retrieved: {len(sources)}")
            
            # Store detailed results
            detailed_results.append({
                "question": test_case['question'],
                "generated_answer": generated_answer,
                "expected_answer": test_case['expected_answer'],
                "similarity_score": similarity_score,
                "response_time": response_time,
                "sources_count": len(sources)
            })
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            detailed_results.append({
                "question": test_case['question'],
                "error": str(e),
                "similarity_score": 0
            })
    
    # Calculate overall performance
    avg_score = total_score / len(test_cases)
    
    print("\n" + "=" * 50)
    print("📈 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"🎯 Overall Average Similarity Score: {avg_score:.2f}/10")
    print(f"📊 Performance Grade: {get_performance_grade(avg_score)}")
    
    # Performance breakdown
    successful_queries = len([r for r in detailed_results if 'error' not in r])
    avg_response_time = sum(r.get('response_time', 0) for r in detailed_results if 'response_time' in r) / max(successful_queries, 1)
    
    print(f"⚡ Average Response Time: {avg_response_time:.2f}s")
    print(f"✅ Successful Queries: {successful_queries}/{len(test_cases)}")
    
    return detailed_results, avg_score

def calculate_similarity(generated_answer, expected_answer):
    """
    Calculate similarity between generated and expected answers
    Uses multiple approaches to measure semantic similarity
    """
    # Convert to lowercase for comparison
    gen_answer = generated_answer.lower()
    exp_answer = expected_answer.lower()
    
    # Method 1: Basic sequence similarity
    basic_similarity = SequenceMatcher(None, gen_answer, exp_answer).ratio()
    
    # Method 2: Key phrase matching
    # Extract key phrases from expected answer
    key_phrases = []
    if "privacy notice" in exp_answer:
        key_phrases.append("privacy notice")
    if "opt out" in exp_answer:
        key_phrases.append("opt out")
    if "arbitrate" in exp_answer:
        key_phrases.append("arbitrate")
    if "marketing" in exp_answer:
        key_phrases.append("marketing")
    if "contact" in exp_answer:
        key_phrases.append("contact")
    if "litigation department" in exp_answer:
        key_phrases.append("litigation department")
    if "583 west ebay way" in exp_answer:
        key_phrases.append("583 west ebay way")
    if "30 days" in exp_answer:
        key_phrases.append("30 days")
    
    # Calculate key phrase coverage
    phrase_matches = sum(1 for phrase in key_phrases if phrase in gen_answer)
    phrase_coverage = phrase_matches / max(len(key_phrases), 1) if key_phrases else 0
    
    # Method 3: Word overlap
    gen_words = set(gen_answer.split())
    exp_words = set(exp_answer.split())
    word_overlap = len(gen_words & exp_words) / max(len(exp_words), 1)
    
    # Combine methods with weights
    final_similarity = (
        basic_similarity * 0.3 +
        phrase_coverage * 0.5 +
        word_overlap * 0.2
    )
    
    # Convert to 0-10 scale
    similarity_score = final_similarity * 10
    
    return similarity_score

def get_performance_grade(score):
    """
    Convert numerical score to letter grade
    """
    if score >= 9:
        return "A+ (Excellent)"
    elif score >= 8:
        return "A (Very Good)"
    elif score >= 7:
        return "B+ (Good)"
    elif score >= 6:
        return "B (Fair)"
    elif score >= 5:
        return "C (Needs Improvement)"
    else:
        return "D (Poor)"

if __name__ == "__main__":
    print("🚀 Starting RAG System Evaluation...")
    results, avg_score = evaluate_rag_system()
    
    print(f"\n🎉 Evaluation Complete! Average Score: {avg_score:.2f}/10")
