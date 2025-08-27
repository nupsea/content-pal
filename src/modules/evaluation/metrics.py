"""
Evaluation metrics for search and retrieval systems
"""

from typing import List, Dict, Any
from collections import defaultdict


def hit_rate_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    """Calculate hit rate at k"""
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    """Calculate Mean Reciprocal Rank at k"""
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id == gold_id:
            return 1.0 / i
    return 0.0


def calculate_metrics(results: List[Dict[str, Any]], k_values: List[int] = None) -> Dict[str, float]:
    """
    Calculate comprehensive metrics from evaluation results
    
    Args:
        results: List of evaluation results with 'ranked_ids' and 'gold_id'
        k_values: List of k values to evaluate (default: [1, 5, 10])
    
    Returns:
        Dictionary of metric names to values
    """
    if k_values is None:
        k_values = [1, 5, 10]
    
    if not results:
        return {f"hit_rate_at_{k}": 0.0 for k in k_values} | {f"mrr_at_{k}": 0.0 for k in k_values}
    
    metrics = defaultdict(list)
    
    for result in results:
        ranked_ids = result.get('ranked_ids', [])
        gold_id = result.get('gold_id', '')
        
        if not ranked_ids or not gold_id:
            continue
        
        for k in k_values:
            metrics[f"hit_rate_at_{k}"].append(hit_rate_at_k(ranked_ids, gold_id, k))
            metrics[f"mrr_at_{k}"].append(mrr_at_k(ranked_ids, gold_id, k))
    
    # Calculate averages
    avg_metrics = {}
    for metric_name, values in metrics.items():
        avg_metrics[metric_name] = sum(values) / len(values) if values else 0.0
    
    # Add total queries
    avg_metrics['total_queries'] = len(results)
    
    return avg_metrics


def calculate_metrics_by_category(results: List[Dict[str, Any]], 
                                  category_key: str = 'intent_type',
                                  k_values: List[int] = None) -> Dict[str, Dict[str, float]]:
    """Calculate metrics broken down by category"""
    if k_values is None:
        k_values = [1, 5, 10]
    
    # Group results by category
    categorized = defaultdict(list)
    for result in results:
        category = result.get(category_key, 'unknown')
        categorized[category].append(result)
    
    # Calculate metrics for each category
    category_metrics = {}
    for category, cat_results in categorized.items():
        category_metrics[category] = calculate_metrics(cat_results, k_values)
    
    return category_metrics