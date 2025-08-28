#!/usr/bin/env python3
"""
Enhanced evaluation script for the adaptive retrieval system.
Compares performance across different strategies and provides detailed analysis.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from tqdm.auto import tqdm
from opensearchpy import OpenSearch
from search.index_assets import make_client
# Define metrics locally to avoid import issues
def hit_rate_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0

def mrr_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    for i, sid in enumerate(ranked_ids[:k], start=1):
        if sid == gold_id:
            return 1.0 / i
    return 0.0
from rag.adaptive_retrieval import AdaptiveRetriever
from rag.query_classifier import QueryClassifier


def evaluate_adaptive_system(
    client: OpenSearch,
    index: str, 
    qid_to_queries: Dict[str, List[str]],
    top_k: int = 10,
    max_pairs: int = 0,
    use_reranking: bool = False,
    workers: int = 8
) -> Dict:
    """
    Evaluate the adaptive retrieval system with detailed breakdowns.
    """
    
    # Prepare evaluation pairs
    pairs = []
    for show_id, queries in qid_to_queries.items():
        for query in queries:
            if query and isinstance(query, str):
                pairs.append((show_id, query.strip()))
    
    if max_pairs and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    
    print(f"Evaluating {len(pairs)} query-document pairs...")
    
    # Initialize retriever and classifier
    retriever = AdaptiveRetriever(client=client, index=index)
    
    # Storage for results
    results_by_intent = defaultdict(list)
    overall_results = []
    intent_counts = Counter()
    
    def evaluate_single_pair(gold_id: str, query: str) -> Dict:
        """Evaluate a single query-document pair"""
        try:
            # Get retrieval results
            result = retriever.retrieve(query, top_k=top_k*2, use_reranking=use_reranking)
            
            # Extract document IDs from hits
            hits = result.get("hits", [])
            retrieved_ids = []
            for hit in hits:
                source = hit.get("_source", {})
                doc_id = source.get("show_id", "") or hit.get("_id", "")
                if doc_id:
                    retrieved_ids.append(doc_id)
            
            # Calculate metrics
            hr = hit_rate_at_k(retrieved_ids, gold_id, k=top_k)
            mrr = mrr_at_k(retrieved_ids, gold_id, k=top_k)
            
            # Get intent information
            intent = result.get("query_intent")
            intent_type = intent.intent_type if intent else "UNKNOWN"
            
            return {
                "gold_id": gold_id,
                "query": query,
                "hit_rate": hr,
                "mrr": mrr,
                "intent_type": intent_type,
                "strategy_used": result.get("strategy_used", "unknown"),
                "confidence": intent.confidence if intent else 0.0,
                "num_hits": len(hits),
                "reranked": result.get("reranked", False)
            }
            
        except Exception as e:
            print(f"Error evaluating query '{query}': {e}")
            return {}
    
    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(evaluate_single_pair, gold_id, query)
            for gold_id, query in pairs
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = future.result()
            if result:
                overall_results.append(result)
                intent_type = result["intent_type"]
                results_by_intent[intent_type].append(result)
                intent_counts[intent_type] += 1
    
    # Calculate overall metrics
    total_pairs = len(overall_results)
    overall_hr = sum(r["hit_rate"] for r in overall_results) / max(1, total_pairs)
    overall_mrr = sum(r["mrr"] for r in overall_results) / max(1, total_pairs)
    
    # Calculate metrics by intent type
    metrics_by_intent = {}
    for intent_type, results in results_by_intent.items():
        if results:
            intent_hr = sum(r["hit_rate"] for r in results) / len(results)
            intent_mrr = sum(r["mrr"] for r in results) / len(results)
            avg_confidence = sum(r["confidence"] for r in results) / len(results)
            
            metrics_by_intent[intent_type] = {
                "count": len(results),
                "hit_rate": intent_hr,
                "mrr": intent_mrr,
                "avg_confidence": avg_confidence,
                "examples": results[:3]  # Keep a few examples
            }
    
    return {
        "overall": {
            "total_pairs": total_pairs,
            "hit_rate": overall_hr,
            "mrr": overall_mrr,
            "top_k": top_k
        },
        "by_intent": metrics_by_intent,
        "intent_distribution": dict(intent_counts),
        "detailed_results": overall_results
    }


def compare_strategies(results: Dict) -> None:
    """Compare performance across different intent types and strategies"""
    
    print("\n" + "=" * 70)
    print("ADAPTIVE RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    
    overall = results["overall"]
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total Pairs: {overall['total_pairs']}")
    print(f"  Hit Rate@{overall['top_k']}: {overall['hit_rate']:.4f}")
    print(f"  MRR@{overall['top_k']}: {overall['mrr']:.4f}")
    
    print(f"\nINTENT TYPE DISTRIBUTION:")
    intent_dist = results["intent_distribution"]
    total = sum(intent_dist.values())
    for intent, count in sorted(intent_dist.items()):
        percentage = (count / total) * 100
        print(f"  {intent:12}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nPERFORMANCE BY INTENT TYPE:")
    print(f"{'Intent':<12} {'Count':<6} {'Hit Rate':<10} {'MRR':<10} {'Confidence':<11}")
    print("-" * 60)
    
    by_intent = results["by_intent"]
    for intent in sorted(by_intent.keys()):
        metrics = by_intent[intent]
        print(f"{intent:<12} {metrics['count']:<6} {metrics['hit_rate']:<10.4f} "
              f"{metrics['mrr']:<10.4f} {metrics['avg_confidence']:<11.3f}")
    
    # Find best and worst performing intent types
    intent_performance = [(intent, metrics['mrr']) for intent, metrics in by_intent.items()]
    intent_performance.sort(key=lambda x: x[1], reverse=True)
    
    if intent_performance:
        print(f"\nBEST PERFORMING INTENT: {intent_performance[0][0]} (MRR: {intent_performance[0][1]:.4f})")
        print(f"WORST PERFORMING INTENT: {intent_performance[-1][0]} (MRR: {intent_performance[-1][1]:.4f})")
    
    # Show examples of failed queries (hit_rate = 0)
    print(f"\nFAILED QUERY EXAMPLES (Hit Rate = 0):")
    failed_queries = [r for r in results["detailed_results"] if r["hit_rate"] == 0.0]
    failed_by_intent = defaultdict(list)
    for failure in failed_queries:
        failed_by_intent[failure["intent_type"]].append(failure)
    
    for intent, failures in failed_by_intent.items():
        print(f"\n{intent} ({len(failures)} failures):")
        for failure in failures[:3]:  # Show first 3
            print(f"  Query: '{failure['query'][:60]}{'...' if len(failure['query']) > 60 else ''}'")
            print(f"    Strategy: {failure['strategy_used']}, Confidence: {failure['confidence']:.2f}")
    
    print(f"\nOVERALL FAILURE RATE: {len(failed_queries)}/{overall['total_pairs']} "
          f"({len(failed_queries)/max(1, overall['total_pairs'])*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptive retrieval system")
    parser.add_argument("--index", default="netflix_assets_v6", help="OpenSearch index name")
    parser.add_argument("--pairs", default="../../notebooks/ground_truth.json", 
                       help="Path to ground truth pairs")
    parser.add_argument("--top_k", type=int, default=10, help="Evaluation top-k")
    parser.add_argument("--max_pairs", type=int, help="Limit evaluation to N pairs")
    parser.add_argument("--use_reranking", action="store_true", help="Use cross-encoder reranking")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output", help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    # Load ground truth
    try:
        with open(args.pairs, 'r') as f:
            qid_to_queries = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {args.pairs}")
        print("Please run from src/rag/ directory or provide correct path")
        return
    
    # Run evaluation
    client = make_client()
    results = evaluate_adaptive_system(
        client=client,
        index=args.index,
        qid_to_queries=qid_to_queries,
        top_k=args.top_k,
        max_pairs=args.max_pairs,
        use_reranking=args.use_reranking,
        workers=args.workers
    )
    
    # Display results
    compare_strategies(results)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()