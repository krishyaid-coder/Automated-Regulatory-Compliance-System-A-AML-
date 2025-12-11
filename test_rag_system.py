"""
Testing Framework for RAG System
Simple functions for testing citation accuracy and hallucination rate
"""

import json
import os
from typing import List, Dict, Any
from rag_query_system import query


def get_default_questions() -> List[Dict[str, Any]]:
    """Get default test questions."""
    return [
        {
            "question": "What are the customer due diligence requirements?",
            "expected_keywords": ["CDD", "customer", "due diligence", "identification"],
            "category": "CDD Requirements"
        },
        {
            "question": "What is the threshold for reporting suspicious transactions?",
            "expected_keywords": ["threshold", "suspicious", "transaction", "reporting"],
            "category": "Transaction Reporting"
        },
        {
            "question": "What information is required for wire transfers?",
            "expected_keywords": ["wire transfer", "originator", "beneficiary", "information"],
            "category": "Wire Transfers"
        },
        {
            "question": "What are the KYC requirements for opening an account?",
            "expected_keywords": ["KYC", "know your customer", "account", "identification"],
            "category": "KYC Requirements"
        },
        {
            "question": "What is required for AML compliance programs?",
            "expected_keywords": ["AML", "compliance", "program", "policies"],
            "category": "AML Programs"
        }
    ]


def load_test_questions(questions_file='test_questions.json') -> List[Dict[str, Any]]:
    """Load test questions from JSON file."""
    if os.path.exists(questions_file):
        with open(questions_file, 'r') as f:
            return json.load(f)
    else:
        return get_default_questions()


def evaluate_citations(citations: List[Dict]) -> Dict[str, Any]:
    """Evaluate citation quality."""
    if not citations:
        return {
            'score': 0.0,
            'has_source_document': False,
            'has_section': False,
            'avg_similarity': 0.0
        }
    
    has_source = all('filename' in c and c['filename'] != 'Unknown' for c in citations)
    has_section = any('Section' in c.get('source', '') for c in citations)
    avg_similarity = sum(c.get('similarity_score', 0) for c in citations) / len(citations)
    
    score = 0.0
    if has_source:
        score += 0.4
    if has_section:
        score += 0.3
    if avg_similarity > 0.5:
        score += 0.3
    
    return {
        'score': score,
        'has_source_document': has_source,
        'has_section': has_section,
        'avg_similarity': avg_similarity
    }


def check_hallucination(answer: str, citations: List[Dict]) -> Dict[str, Any]:
    """Check for potential hallucination indicators."""
    uncertainty_phrases = [
        "i don't know",
        "i cannot find",
        "not in the provided",
        "cannot be determined"
    ]
    
    has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
    has_citation_markers = any(f"[Citation {i}]" in answer or f"[{i}]" in answer 
                              for i in range(1, len(citations) + 1))
    is_too_short = len(answer) < 50
    
    generic_responses = [
        "based on the information",
        "according to the document",
        "the text states"
    ]
    is_generic = any(phrase in answer.lower() for phrase in generic_responses) and not has_citation_markers
    
    return {
        'has_uncertainty_phrases': has_uncertainty,
        'has_citation_markers': has_citation_markers,
        'is_too_short': is_too_short,
        'is_generic': is_generic,
        'hallucination_risk': 'low' if (has_citation_markers and not is_generic) else 'medium'
    }


def test_question(question_data: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
    """Test a single question."""
    question = question_data['question']
    expected_keywords = question_data.get('expected_keywords', [])
    category = question_data.get('category', 'General')
    
    # Query RAG system
    result = query(question, use_llm=use_llm)
    
    # Evaluate results
    answer = result['answer'].lower()
    citations = result['citations']
    confidence = result['confidence']
    
    # Check for expected keywords
    keywords_found = [kw for kw in expected_keywords if kw.lower() in answer]
    keyword_coverage = len(keywords_found) / len(expected_keywords) if expected_keywords else 0
    
    # Check citation quality
    has_citations = len(citations) > 0
    citation_quality = evaluate_citations(citations)
    
    # Check for hallucination indicators
    hallucination_indicators = check_hallucination(answer, citations)
    
    return {
        'question': question,
        'category': category,
        'answer': result['answer'],
        'citations': citations,
        'num_citations': len(citations),
        'has_citations': has_citations,
        'citation_quality': citation_quality,
        'confidence': confidence,
        'keywords_found': keywords_found,
        'keyword_coverage': keyword_coverage,
        'hallucination_indicators': hallucination_indicators,
        'retrieved_chunks': len(result['retrieved_chunks'])
    }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics from test results."""
    total = len(results)
    
    has_citations_count = sum(1 for r in results if r['has_citations'])
    citation_rate = has_citations_count / total if total > 0 else 0
    avg_citation_quality = sum(r['citation_quality']['score'] for r in results) / total if total > 0 else 0
    avg_keyword_coverage = sum(r['keyword_coverage'] for r in results) / total if total > 0 else 0
    avg_confidence = sum(r['confidence'] for r in results) / total if total > 0 else 0
    
    hallucination_risks = [r['hallucination_indicators']['hallucination_risk'] for r in results]
    low_risk_count = sum(1 for r in hallucination_risks if r == 'low')
    hallucination_mitigation_rate = low_risk_count / total if total > 0 else 0
    
    return {
        'citation_rate': citation_rate,
        'avg_citation_quality': avg_citation_quality,
        'avg_keyword_coverage': avg_keyword_coverage,
        'avg_confidence': avg_confidence,
        'hallucination_mitigation_rate': hallucination_mitigation_rate,
        'total_questions': total
    }


def run_test_suite(questions: List[Dict[str, Any]], use_llm: bool = True) -> Dict[str, Any]:
    """Run full test suite."""
    print("="*80)
    print("Running RAG Test Suite")
    print("="*80)
    
    results = []
    for i, question_data in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Testing: {question_data['question']}")
        result = test_question(question_data, use_llm=use_llm)
        results.append(result)
    
    metrics = calculate_metrics(results)
    
    return {
        'results': results,
        'metrics': metrics,
        'total_questions': len(questions)
    }


def print_report(test_results: Dict[str, Any]):
    """Print test report."""
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    metrics = test_results['metrics']
    print(f"\nTotal Questions Tested: {metrics['total_questions']}")
    print(f"\nCitation Metrics:")
    print(f"  Citation Rate: {metrics['citation_rate']:.1%}")
    print(f"  Average Citation Quality: {metrics['avg_citation_quality']:.2f}/1.0")
    print(f"\nAnswer Quality:")
    print(f"  Average Keyword Coverage: {metrics['avg_keyword_coverage']:.1%}")
    print(f"  Average Confidence: {metrics['avg_confidence']:.2f}")
    print(f"\nHallucination Mitigation:")
    print(f"  Low Risk Rate: {metrics['hallucination_mitigation_rate']:.1%}")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for i, result in enumerate(test_results['results'], 1):
        print(f"\n[{i}] {result['question']}")
        print(f"    Category: {result['category']}")
        print(f"    Citations: {result['num_citations']} (Quality: {result['citation_quality']['score']:.2f})")
        print(f"    Keyword Coverage: {result['keyword_coverage']:.1%}")
        print(f"    Confidence: {result['confidence']:.3f}")
        print(f"    Hallucination Risk: {result['hallucination_indicators']['hallucination_risk']}")


def main():
    """Run test suite."""
    print("="*80)
    print("RAG System Testing Framework")
    print("="*80)
    
    # Initialize system
    from rag_query_system import initialize_system
    try:
        initialize_system()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load test questions
    questions = load_test_questions()
    
    # Run tests
    results = run_test_suite(questions, use_llm=False)
    
    # Print report
    print_report(results)
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Test results saved to test_results.json")


if __name__ == "__main__":
    main()
