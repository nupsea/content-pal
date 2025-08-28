"""
Comprehensive evaluation system for search backends and RAG systems
"""

import json
import time
from typing import Dict, List, Any, Optional, Union, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from tqdm import tqdm

from .metrics import calculate_metrics, calculate_metrics_by_category
from ..search import ContentSearchSystem, SearchConfig
from ..rag import AdaptiveRetriever


class SearchEvaluator:
    """Comprehensive evaluator for search systems"""
    
    def __init__(self, search_system: Union[ContentSearchSystem, AdaptiveRetriever]):
        """Initialize with search system or adaptive retriever"""
        self.system = search_system
        self.is_adaptive = isinstance(search_system, AdaptiveRetriever)
    
    def evaluate_single_query(self, query: str, gold_id: str, **kwargs) -> Dict[str, Any]:
        """Evaluate a single query"""
        try:
            start_time = time.time()
            
            if self.is_adaptive:
                # Use adaptive retriever (cast for type checking)
                adaptive_system = cast(AdaptiveRetriever, self.system)
                result = adaptive_system.retrieve(query, **kwargs)
                hits = result.get('hits', [])
                intent_info = result.get('query_intent')
                strategy = result.get('strategy_used', 'unknown')
            else:
                # Use basic search system
                config = kwargs.get('config') or SearchConfig(
                    boost_weights={'title': 2.0, 'cast': 4.0, 'director': 2.0, 
                                 'listed_in': 1.0, 'description': 0.5},
                    max_results=kwargs.get('top_k', 50)
                )
                # Cast to ContentSearchSystem to satisfy static type checking
                search_system = cast(ContentSearchSystem, self.system)
                search_results = search_system.search(query, config)
                hits = [{'_source': {'show_id': r.id}} for r in search_results]
                intent_info = None
                strategy = 'basic_search'
            
            query_time = time.time() - start_time
            
            # Extract ranked IDs
            ranked_ids = []
            for hit in hits:
                source = hit.get('_source', {})
                doc_id = source.get('show_id') or source.get('id') or hit.get('_id', '')
                if doc_id:
                    ranked_ids.append(doc_id)
            
            return {
                'query': query,
                'gold_id': gold_id,
                'ranked_ids': ranked_ids,
                'num_results': len(ranked_ids),
                'query_time': query_time,
                'strategy_used': strategy,
                'intent_type': intent_info.intent_type.value if intent_info else 'unknown',
                'intent_confidence': intent_info.confidence if intent_info else 0.0,
                'success': True
            }
            
        except Exception as e:
            return {
                'query': query,
                'gold_id': gold_id,
                'ranked_ids': [],
                'num_results': 0,
                'query_time': 0.0,
                'strategy_used': 'error',
                'intent_type': 'error',
                'intent_confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_queries(self, queries: Dict[str, List[str]], 
                        max_queries: Optional[int] = None,
                        workers: int = 4, **kwargs) -> Dict[str, Any]:
        """
        Evaluate multiple queries
        
        Args:
            queries: Dict mapping gold_id -> list of queries
            max_queries: Limit total number of queries to evaluate
            workers: Number of parallel workers
            **kwargs: Arguments passed to search system
        
        Returns:
            Comprehensive evaluation results
        """
        # Prepare query pairs
        query_pairs = []
        for gold_id, query_list in queries.items():
            for query in query_list:
                if query and isinstance(query, str):
                    query_pairs.append((gold_id, query.strip()))
        
        if max_queries and len(query_pairs) > max_queries:
            query_pairs = query_pairs[:max_queries]
        
        
        # Parallel evaluation
        results = []
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.evaluate_single_query, query, gold_id, **kwargs)
                    for gold_id, query in query_pairs
                ]
                
                iterator = tqdm(as_completed(futures), total=len(futures), desc="Evaluating")
                
                for future in iterator:
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            # Sequential evaluation
            iterator = tqdm(query_pairs, desc="Evaluating")
            
            for gold_id, query in iterator:
                result = self.evaluate_single_query(query, gold_id, **kwargs)
                if result:
                    results.append(result)
        
        return self.compile_results(results)
    
    def compile_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile evaluation results into comprehensive report"""
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        # Calculate overall metrics
        overall_metrics = calculate_metrics(successful_results, k_values=[1, 5, 10])
        
        # Calculate metrics by intent type
        intent_metrics = calculate_metrics_by_category(
            successful_results, 'intent_type', k_values=[1, 5, 10]
        )
        
        # Calculate metrics by strategy
        strategy_metrics = calculate_metrics_by_category(
            successful_results, 'strategy_used', k_values=[1, 5, 10]
        )
        
        # Performance statistics
        query_times = [r.get('query_time', 0) for r in successful_results]
        avg_query_time = sum(query_times) / len(query_times) if query_times else 0
        
        # Intent distribution
        intent_distribution = defaultdict(int)
        for result in successful_results:
            intent_distribution[result.get('intent_type', 'unknown')] += 1
        
        # Strategy distribution
        strategy_distribution = defaultdict(int)
        for result in successful_results:
            strategy_distribution[result.get('strategy_used', 'unknown')] += 1
        
        # Failed query examples
        failed_examples = failed_results[:10]  # First 10 failures
        
        # Zero-result queries
        zero_results = [r for r in successful_results if r.get('num_results', 0) == 0]
        
        return {
            'summary': {
                'total_queries': len(results),
                'successful_queries': len(successful_results),
                'failed_queries': len(failed_results),
                'zero_result_queries': len(zero_results),
                'avg_query_time_ms': avg_query_time * 1000,
                'system_type': 'adaptive' if self.is_adaptive else 'basic'
            },
            'overall_metrics': overall_metrics,
            'metrics_by_intent': dict(intent_metrics),
            'metrics_by_strategy': dict(strategy_metrics),
            'distributions': {
                'intent_types': dict(intent_distribution),
                'strategies': dict(strategy_distribution)
            },
            'failed_examples': failed_examples,
            'zero_result_examples': zero_results[:10],
            'detailed_results': successful_results
        }
    
    def compare_configurations(self, queries: Dict[str, List[str]], 
                             configs: List[Dict[str, Any]], 
                             config_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple search configurations"""
        if config_names is None:
            config_names = [f"config_{i+1}" for i in range(len(configs))]
        
        comparison_results = {}
        
        for i, config in enumerate(configs):
            config_name = config_names[i]
            print(f"\nEvaluating configuration: {config_name}")
            
            result = self.evaluate_queries(queries, **config)
            comparison_results[config_name] = result
        
        # Create comparison summary
        summary = {}
        for metric in ['hit_rate_at_1', 'hit_rate_at_5', 'hit_rate_at_10', 'mrr_at_10']:
            summary[metric] = {
                name: results['overall_metrics'].get(metric, 0.0)
                for name, results in comparison_results.items()
            }
        
        return {
            'individual_results': comparison_results,
            'comparison_summary': summary,
            'best_config': max(summary['mrr_at_10'], key=summary['mrr_at_10'].get)
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON file"""
        # Make results JSON serializable
        def serialize_obj(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=serialize_obj)
        print(f"Results saved to: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        summary = results['summary']
        overall = results['overall_metrics']
        
        print(f"\nSystem: {summary['system_type']}")
        print(f"Total queries: {summary['total_queries']:,}")
        print(f"Successful: {summary['successful_queries']:,}")
        print(f"Failed: {summary['failed_queries']:,}")
        print(f"Zero results: {summary['zero_result_queries']:,}")
        print(f"Avg query time: {summary['avg_query_time_ms']:.1f}ms")
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Hit Rate@1:  {overall.get('hit_rate_at_1', 0):.4f}")
        print(f"  Hit Rate@5:  {overall.get('hit_rate_at_5', 0):.4f}")
        print(f"  Hit Rate@10: {overall.get('hit_rate_at_10', 0):.4f}")
        print(f"  MRR@10:      {overall.get('mrr_at_10', 0):.4f}")
        
        # Intent breakdown if available
        if results.get('metrics_by_intent'):
            print(f"\nPERFORMANCE BY INTENT:")
            print(f"{'Intent':<15} {'Count':<6} {'HR@10':<8} {'MRR@10':<8}")
            print("-" * 40)
            
            for intent, metrics in results['metrics_by_intent'].items():
                count = int(metrics.get('total_queries', 0))
                hr10 = metrics.get('hit_rate_at_10', 0)
                mrr10 = metrics.get('mrr_at_10', 0)
                print(f"{intent:<15} {count:<6} {hr10:<8.4f} {mrr10:<8.4f}")
        
        print("\n" + "="*70)