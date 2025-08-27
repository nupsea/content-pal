#!/usr/bin/env python3
"""
Comprehensive evaluation using the new modular system
"""

import sys
import json
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.search import ContentSearchSystem
from modules.rag import AdaptiveRetriever
from modules.evaluation import SearchEvaluator, GroundTruthGenerator


def run_comprehensive_evaluation():
    """Run comprehensive evaluation with multiple configurations"""
    
    print("COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    # Load or generate ground truth
    gt_file = "../comprehensive_ground_truth.json"
    if Path(gt_file).exists():
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)
    else:
        generator = GroundTruthGenerator()
        ground_truth = generator.generate_ground_truth(
            csv_path="data/netflix_titles_cleaned.csv",
            output_path="comprehensive_ground_truth.json",
            sample_size=1000
        )
    
    generator = GroundTruthGenerator()
    eval_subset = generator.create_evaluation_subset(ground_truth, subset_size=200)
    
    # Step 2: Test different configurations
    configurations = [
        {
            "name": "MinSearch_Default",
            "system_type": "basic",
            "backend": "minsearch",
            "config": {
                "boost_weights": {"title": 3.0, "cast": 2.0, "director": 1.5, 
                                "listed_in": 1.5, "description": 1.0}
            }
        },
        {
            "name": "MinSearch_Optimized", 
            "system_type": "basic",
            "backend": "minsearch",
            "config": {
                "boost_weights": {"title": 4.0, "cast": 3.5, "director": 2.5,
                                "listed_in": 2.0, "description": 1.5}
            }
        },
        {
            "name": "Adaptive_Default",
            "system_type": "adaptive",
            "backend": "minsearch"
        },
        {
            "name": "OpenSearch_Hybrid",
            "system_type": "adaptive", 
            "backend": "opensearch"
        }
    ]
    
    
    results = {}
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        try:
            
            # Create system based on configuration
            if config["system_type"] == "basic":
                system = ContentSearchSystem(backend_type=config["backend"])
                system.index_data(csv_path="data/netflix_titles_cleaned.csv")
                evaluator = SearchEvaluator(system)
                
                # Use custom config if provided
                search_config = None
                if "config" in config:
                    from modules.search import SearchConfig
                    search_config = SearchConfig(**config["config"], max_results=50)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=500,  # Limit for reasonable runtime
                    workers=4,
                    config=search_config
                )
                
            else:  # adaptive
                system = AdaptiveRetriever(backend_type=config["backend"])
                system.index_data(csv_path="data/netflix_titles_cleaned.csv")
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=500,
                    workers=4,
                    top_k=50,
                    use_reranking=False  # Set to True if you want reranking
                )
            
            results[config["name"]] = eval_results
            
            # Print quick summary
            metrics = eval_results["overall_metrics"]
            print(f"{config['name']} - HR@10: {metrics.get('hit_rate_at_10', 0):.4f}, MRR@10: {metrics.get('mrr_at_10', 0):.4f}, Time: {eval_results['summary']['avg_query_time_ms']:.1f}ms")
            
        except ImportError as e:
            print(f"IMPORT ERROR: {config['name']}: {e}")
            results[config["name"]] = {"error": f"Import error: {e}"}
        except Exception as e:
            print(f"RUNTIME ERROR: {config['name']}: {e}")
            results[config["name"]] = {"error": str(e)}
    
    # Step 3: Compare results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Configuration':<20} {'HR@1':<8} {'HR@5':<8} {'HR@10':<8} {'MRR@10':<8} {'Avg Time':<10}")
    print("-" * 80)
    
    best_config = None
    best_mrr = 0.0
    
    for name, result in results.items():
        if "error" not in result:
            metrics = result["overall_metrics"]
            summary = result["summary"]
            
            hr1 = metrics.get('hit_rate_at_1', 0)
            hr5 = metrics.get('hit_rate_at_5', 0) 
            hr10 = metrics.get('hit_rate_at_10', 0)
            mrr = metrics.get('mrr_at_10', 0)
            time_ms = summary['avg_query_time_ms']
            
            print(f"{name:<20} {hr1:<8.4f} {hr5:<8.4f} {hr10:<8.4f} {mrr:<8.4f} {time_ms:<10.1f}")
            
            if mrr > best_mrr:
                best_mrr = mrr
                best_config = name
        else:
            print(f"{name:<20} ERROR: {result['error']}")
    
    if best_config:
        print(f"\nBEST CONFIGURATION: {best_config} (MRR@10: {best_mrr:.4f})")
    
    # Step 4: Detailed analysis for best configuration
    if best_config and best_config in results:
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS: {best_config}")
        print(f"{'='*80}")
        
        best_result = results[best_config]
        
        # Show performance by intent type if available
        if "metrics_by_intent" in best_result:
            print(f"\nPerformance by Intent Type:")
            print(f"{'Intent':<15} {'Count':<6} {'HR@10':<8} {'MRR@10':<8}")
            print("-" * 40)
            
            for intent, metrics in best_result["metrics_by_intent"].items():
                count = int(metrics.get('total_queries', 0))
                hr10 = metrics.get('hit_rate_at_10', 0)
                mrr10 = metrics.get('mrr_at_10', 0)
                print(f"{intent:<15} {count:<6} {hr10:<8.4f} {mrr10:<8.4f}")
        
        # Show failed query examples
        failed_examples = best_result.get("failed_examples", [])
        if failed_examples:
            print(f"\nFailed Query Examples:")
            for example in failed_examples[:5]:
                print(f"  '{example.get('query', '')}': {example.get('error', '')}")
    
    # Step 5: Save results
    output_file = "comprehensive_evaluation_results.json"
    
    # Make results serializable
    def serialize_obj(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=serialize_obj)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_evaluation()