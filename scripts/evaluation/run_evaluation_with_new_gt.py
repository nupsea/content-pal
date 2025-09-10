#!/usr/bin/env python3
"""
Run comprehensive evaluation with the new independent ground truth (ground_truth_2.json)
"""

import sys
import json
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "src"))

from modules.search import ContentSearchSystem
from modules.search.semantic_search import SemanticSearchSystem
from modules.rag_old import AdaptiveRetriever
from modules.evaluation import SearchEvaluator

def run_evaluation_with_new_ground_truth():
    """Run evaluation with the new independent ground truth"""
    
    print("🎯 COMPREHENSIVE EVALUATION WITH NEW GROUND TRUTH")
    print("=" * 70)
    
    # Load the new ground truth directly
    gt_file = "comprehensive_ground_truth.json" 
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"✅ Loaded new ground truth: {len(ground_truth)} unique assets")
    
    # Create evaluation subset (smaller for faster execution)
    import random
    random.seed(42)  # Reproducible results
    
    asset_ids = list(ground_truth.keys())
    random.shuffle(asset_ids)
    subset_size = min(100, len(asset_ids))  # Limit to 100 assets for speed
    eval_subset = {aid: ground_truth[aid] for aid in asset_ids[:subset_size]}
    
    print(f"✅ Created evaluation subset: {subset_size} assets")
    
    # Simplified configurations for faster evaluation
    configurations = [
        {
            "name": "Basic_Default",
            "system_type": "basic",
            "backend": "minsearch",
            "config": {
                "boost_weights": {"title": 3.0, "cast": 2.0, "director": 1.5, 
                                "listed_in": 1.5, "description": 1.0}
            }
        },
        {
            "name": "Basic_Optimized", 
            "system_type": "basic",
            "backend": "minsearch",
            "config": {
                "boost_weights": {"title": 4.0, "cast": 3.5, "director": 2.5,
                                "listed_in": 2.0, "description": 1.5}
            }
        },
        {
            "name": "Adaptive_RuleBased",
            "system_type": "adaptive",
            "backend": "minsearch"
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n🔍 Testing: {config['name']}")
        
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
                    max_queries=len(eval_subset) * 5,  # All queries per asset
                    workers=4,
                    config=search_config
                )
                
            elif config["system_type"] == "adaptive":
                system = AdaptiveRetriever(backend_type=config["backend"])
                system.index_data(csv_path="data/netflix_titles_cleaned.csv")
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=len(eval_subset) * 5,
                    workers=4,
                    top_k=50,
                    use_reranking=False
                )
            
            results[config["name"]] = eval_results
            
            # Print quick summary
            metrics = eval_results["overall_metrics"]
            print(f"📊 {config['name']} Results:")
            print(f"   HR@10: {metrics.get('hit_rate_at_10', 0):.4f}")
            print(f"   MRR@10: {metrics.get('mrr_at_10', 0):.4f}") 
            print(f"   Avg Time: {eval_results['summary']['avg_query_time_ms']:.1f}ms")
            
        except Exception as e:
            print(f"❌ ERROR: {config['name']}: {e}")
            results[config["name"]] = {"error": str(e)}
    
    # Final results comparison
    print(f"\n{'='*70}")
    print("🏆 COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Configuration':<20} {'HR@1':<8} {'HR@5':<8} {'HR@10':<8} {'MRR@10':<8} {'Avg Time':<10}")
    print("-" * 70)
    
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
        print(f"\n🥇 BEST PERFORMING SYSTEM: {best_config} (MRR@10: {best_mrr:.4f})")
    
    # Save results
    output_file = "new_ground_truth_evaluation_results.json"
    
    # Make results serializable
    def serialize_obj(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=serialize_obj)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    run_evaluation_with_new_ground_truth()