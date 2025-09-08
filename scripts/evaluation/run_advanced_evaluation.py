#!/usr/bin/env python3
"""
Advanced Search System Evaluation - Targeting >50% MRR/HR Performance

Tests the new AdvancedSemanticSearchSystem against existing best performers
"""

import sys
import json
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from modules.search.advanced_semantic_search import AdvancedSemanticSearchSystem
from modules.search.pretrained_semantic_search import PretrainedSemanticSearch
from modules.evaluation import SearchEvaluator


def run_advanced_evaluation():
    """Compare Advanced Semantic Search against current best systems"""
    print("🚀 ADVANCED SEMANTIC SEARCH EVALUATION")
    print("Target: >50% HR@10 Performance")
    print("=" * 80)
    
    # Load ground truth
    possible_paths = [
        "new_ground_truth.json",
        "../new_ground_truth.json",
        str(Path(__file__).parent.parent / "new_ground_truth.json")
    ]
    
    gt_file = None
    for path in possible_paths:
        if Path(path).exists():
            gt_file = path
            break
    
    if not gt_file:
        print("❌ Ground truth file not found")
        return
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"✅ Loaded ground truth: {len(ground_truth)} unique assets")
    
    # Use a reasonable subset for comprehensive testing
    import random
    random.seed(42)
    asset_ids = list(ground_truth.keys())
    random.shuffle(asset_ids)
    subset_size = min(200, len(asset_ids))  # Larger subset for better accuracy
    eval_subset = {aid: ground_truth[aid] for aid in asset_ids[:subset_size]}
    
    print(f"✅ Created evaluation subset: {subset_size} assets")
    print()
    
    # Test configurations
    configurations = [
        {
            "name": "AdvancedSemantic_FullStack",
            "description": "Advanced system with all optimizations",
            "system_type": "advanced_semantic",
            "use_reranking": True
        },
        {
            "name": "AdvancedSemantic_NoReranking", 
            "description": "Advanced system without cross-encoder",
            "system_type": "advanced_semantic",
            "use_reranking": False
        },
        {
            "name": "PretrainedSemantic_Baseline",
            "description": "Current best performer (31.8% HR@10)",
            "system_type": "pretrained_semantic"
        }
    ]
    
    results = {}
    csv_path = "data/netflix_titles_enriched_full.csv"
    
    for config in configurations:
        print(f"🧪 Testing: {config['name']}")
        print(f"   📝 {config['description']}")
        
        try:
            start_time = time.time()
            
            if config["system_type"] == "advanced_semantic":
                # Initialize Advanced Semantic Search System
                system = AdvancedSemanticSearchSystem(backend_type="minsearch")
                system.index_data(csv_path=csv_path)
                
                # Create evaluator-compatible wrapper
                class AdvancedSemanticWrapper:
                    def __init__(self, advanced_system, use_reranking=True):
                        self.advanced_system = advanced_system
                        self.use_reranking = use_reranking
                    
                    def search(self, query: str, top_k: int = 50):
                        """Search method compatible with evaluator"""
                        from modules.search.core import SearchResult
                        results = self.advanced_system.search(query, top_k, use_reranking=self.use_reranking)
                        return results
                
                wrapped_system = AdvancedSemanticWrapper(system, config.get("use_reranking", True))
                evaluator = SearchEvaluator(wrapped_system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=200,
                    workers=2,
                    top_k=50
                )
                
            elif config["system_type"] == "pretrained_semantic":
                # Baseline comparison
                system = PretrainedSemanticSearch(backend_type="minsearch")
                system.index_data(csv_path=csv_path)
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=200,
                    workers=2,
                    top_k=50
                )
            
            setup_time = time.time() - start_time
            results[config["name"]] = eval_results
            
            # Print results
            metrics = eval_results["overall_metrics"]
            summary = eval_results["summary"]
            
            hr10 = metrics.get('hit_rate_at_10', 0)
            mrr10 = metrics.get('mrr_at_10', 0)
            avg_time = summary.get('avg_query_time_ms', 0)
            
            print(f"   📊 Results:")
            print(f"      HR@10:  {hr10:.4f} ({hr10*100:.1f}%)")
            print(f"      MRR@10: {mrr10:.4f}")
            print(f"      Avg Query Time: {avg_time:.1f}ms")
            print(f"      Setup Time: {setup_time:.1f}s")
            
            # Performance assessment
            if hr10 >= 0.5:
                print(f"      🎯 TARGET ACHIEVED: {hr10*100:.1f}% >= 50%!")
            elif hr10 >= 0.4:
                print(f"      🔥 Excellent: {hr10*100:.1f}% (close to target)")
            elif hr10 >= 0.32:
                print(f"      ✅ Good: {hr10*100:.1f}% (improvement over baseline)")
            else:
                print(f"      ⚠️  Below baseline: {hr10*100:.1f}%")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[config["name"]] = {"error": str(e)}
            print()
    
    # Final comparison
    print("=" * 80)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print(f"{'System':<35} {'HR@10':<8} {'MRR@10':<8} {'Avg Time':<10} {'Status'}")
    print("-" * 80)
    
    best_hr10 = 0
    best_system = ""
    
    for name, result in results.items():
        if "error" not in result:
            metrics = result["overall_metrics"]
            summary = result["summary"]
            
            hr10 = metrics.get('hit_rate_at_10', 0)
            mrr10 = metrics.get('mrr_at_10', 0)
            time_ms = summary.get('avg_query_time_ms', 0)
            
            if hr10 > best_hr10:
                best_hr10 = hr10
                best_system = name
            
            # Status assessment
            if hr10 >= 0.5:
                status = "🎯 TARGET!"
            elif hr10 >= 0.4:
                status = "🔥 Excellent"
            elif hr10 >= 0.32:
                status = "✅ Good"
            else:
                status = "⚠️  Below"
            
            print(f"{name:<35} {hr10:<8.4f} {mrr10:<8.4f} {time_ms:<10.1f} {status}")
        else:
            print(f"{name:<35} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} ❌ Failed")
    
    print("-" * 80)
    print(f"🏆 BEST PERFORMER: {best_system}")
    print(f"   Performance: {best_hr10*100:.1f}% HR@10")
    
    if best_hr10 >= 0.5:
        print(f"   🎉 SUCCESS: Target of 50% ACHIEVED!")
    else:
        improvement_needed = ((0.5 - best_hr10) / best_hr10) * 100
        print(f"   📈 Improvement needed: {improvement_needed:.1f}% to reach 50% target")
    
    # Save results
    timestamp = int(time.time())
    output_file = f"advanced_evaluation_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


def demonstrate_query_understanding():
    """Demonstrate the advanced query understanding capabilities"""
    print("\n" + "=" * 80)
    print("🧠 ADVANCED QUERY UNDERSTANDING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize system
    system = AdvancedSemanticSearchSystem(backend_type="minsearch")
    
    # Test queries that highlight improvements
    test_queries = [
        "Keanu Reeves movies pre-2005",  # The specific issue mentioned
        "80s action films with explosions",
        "recent Marvel superhero movies",
        "classic romantic comedies from the 90s",
        "dark psychological thrillers after 2010",
        "feel-good family movies with animals"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        explanation = system.get_search_explanation(query)
        
        print(f"   🧠 Understood as: {explanation['understood_as']}")
        print(f"   🎯 Confidence: {explanation['confidence']:.2f}")
        
        if explanation['extracted_filters']['temporal']:
            print(f"   📅 Temporal: {explanation['extracted_filters']['temporal']}")
        
        if explanation['extracted_filters']['structured']:
            structured = explanation['extracted_filters']['structured']
            if 'actors' in structured:
                print(f"   👤 Actors: {structured['actors']}")
            if 'genres' in structured:
                print(f"   🎭 Genres: {structured['genres']}")
            if 'mood' in structured:
                print(f"   😊 Mood: {structured['mood']}")
        
        print(f"   🔧 Search Terms: {explanation['search_terms']}")
        if explanation['expansion_terms']:
            print(f"   📈 Expansion: {explanation['expansion_terms']}")
        
        print(f"   💭 Reasoning: {explanation['reasoning'][:100]}...")


if __name__ == "__main__":
    # Run comprehensive evaluation
    results = run_advanced_evaluation()
    
    # Demonstrate query understanding
    demonstrate_query_understanding()
    
    print("\n🏁 Advanced evaluation completed!")