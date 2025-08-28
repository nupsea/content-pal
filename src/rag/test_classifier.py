#!/usr/bin/env python3
"""
Test the query classifier on ground truth data.
Analyzes query patterns to understand the distribution of intent types.
"""

import json
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rag.query_classifier import QueryClassifier, QueryIntent


def load_ground_truth(filepath: str) -> dict:
    """Load ground truth queries"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_queries(ground_truth: dict) -> list:
    """Extract all unique queries from ground truth"""
    all_queries = []
    for show_id, queries in ground_truth.items():
        for query in queries:
            if isinstance(query, str) and len(query.strip()) > 0:
                all_queries.append(query.strip())
    
    return list(set(all_queries))  # Remove duplicates


def classify_and_analyze(queries: list, use_llm: bool = True) -> dict:
    """Classify all queries and analyze patterns"""
    classifier = QueryClassifier(use_llm=use_llm)
    
    results = []
    intent_counts = Counter()
    strategy_counts = Counter()
    confidence_scores = []
    
    print(f"Classifying {len(queries)} unique queries...")
    print("=" * 60)
    
    for i, query in enumerate(queries):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(queries)} queries processed")
        
        intent = classifier.classify(query)
        
        results.append({
            'query': query,
            'intent': intent
        })
        
        intent_counts[intent.intent_type] += 1
        strategy_counts[intent.search_strategy] += 1
        confidence_scores.append(intent.confidence)
    
    # Calculate statistics
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    
    analysis = {
        'total_queries': len(queries),
        'intent_distribution': dict(intent_counts),
        'strategy_distribution': dict(strategy_counts),
        'avg_confidence': avg_confidence,
        'low_confidence_queries': [
            r for r in results if r['intent'].confidence < 0.6
        ],
        'results': results
    }
    
    return analysis


def print_analysis(analysis: dict):
    """Print detailed analysis results"""
    print("\n" + "=" * 60)
    print("QUERY CLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal Queries: {analysis['total_queries']}")
    print(f"Average Confidence: {analysis['avg_confidence']:.3f}")
    
    print(f"\nIntent Distribution:")
    for intent, count in sorted(analysis['intent_distribution'].items()):
        percentage = (count / analysis['total_queries']) * 100
        print(f"  {intent:12}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\nSearch Strategy Distribution:")
    for strategy, count in sorted(analysis['strategy_distribution'].items()):
        percentage = (count / analysis['total_queries']) * 100
        print(f"  {strategy:12}: {count:3d} ({percentage:5.1f}%)")
    
    print(f"\nLow Confidence Queries ({len(analysis['low_confidence_queries'])} total):")
    for item in analysis['low_confidence_queries'][:10]:  # Show first 10
        intent = item['intent']
        print(f"  '{item['query'][:50]}...' -> {intent.intent_type} ({intent.confidence:.2f})")
    
    if len(analysis['low_confidence_queries']) > 10:
        print(f"  ... and {len(analysis['low_confidence_queries']) - 10} more")


def show_examples_by_intent(analysis: dict):
    """Show example queries for each intent type"""
    print("\n" + "=" * 60)
    print("EXAMPLE QUERIES BY INTENT TYPE")
    print("=" * 60)
    
    examples_by_intent = defaultdict(list)
    for result in analysis['results']:
        intent_type = result['intent'].intent_type
        examples_by_intent[intent_type].append(result)
    
    for intent_type in sorted(examples_by_intent.keys()):
        examples = examples_by_intent[intent_type][:5]  # Show top 5 examples
        print(f"\n{intent_type} ({len(examples_by_intent[intent_type])} total):")
        
        for example in examples:
            intent = example['intent']
            print(f"  Query: '{example['query']}'")
            print(f"    Strategy: {intent.search_strategy}")
            print(f"    Confidence: {intent.confidence:.2f}")
            if intent.entities:
                print(f"    Entities: {intent.entities}")
            if intent.filters:
                print(f"    Filters: {intent.filters}")
            print(f"    Reasoning: {intent.explanation}")
            print()


def main():
    """Main execution"""
    # Load ground truth
    ground_truth_path = "../../notebooks/ground_truth.json"
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        print("Please run this script from src/rag/ directory")
        sys.exit(1)
    
    ground_truth = load_ground_truth(ground_truth_path)
    queries = analyze_queries(ground_truth)
    
    # Test with LLM first (if available)
    print("Testing with LLM-based classification...")
    analysis_llm = classify_and_analyze(queries[:50], use_llm=True)  # Test on first 50 for speed
    print_analysis(analysis_llm)
    show_examples_by_intent(analysis_llm)
    
    print("\n" + "=" * 60)
    print("COMPARING WITH RULE-BASED CLASSIFICATION")
    print("=" * 60)
    
    # Test with rule-based classification
    analysis_rules = classify_and_analyze(queries[:50], use_llm=False)
    
    print(f"\nComparison (first 50 queries):")
    print(f"LLM Average Confidence: {analysis_llm['avg_confidence']:.3f}")
    print(f"Rules Average Confidence: {analysis_rules['avg_confidence']:.3f}")
    
    print(f"\nIntent Distribution Comparison:")
    all_intents = set(analysis_llm['intent_distribution'].keys()) | set(analysis_rules['intent_distribution'].keys())
    print(f"{'Intent':<12} {'LLM':<8} {'Rules':<8}")
    print("-" * 30)
    for intent in sorted(all_intents):
        llm_count = analysis_llm['intent_distribution'].get(intent, 0)
        rules_count = analysis_rules['intent_distribution'].get(intent, 0)
        print(f"{intent:<12} {llm_count:<8} {rules_count:<8}")
    
    # Save results for further analysis
    output_file = "classification_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'ground_truth_stats': {
                'total_shows': len(ground_truth),
                'total_unique_queries': len(queries)
            },
            'llm_analysis': analysis_llm,
            'rules_analysis': analysis_rules
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()