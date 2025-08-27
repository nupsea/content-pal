#!/usr/bin/env python3
"""
Quick configuration comparison using existing ground truth
"""

import sys
import json
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.search import ContentSearchSystem, SearchConfig
from modules.evaluation import SearchEvaluator


def quick_comparison():
    """Quick comparison of different boost weight configurations"""
    
    print("QUICK CONFIGURATION COMPARISON")
    print("=" * 50)
    
    gt_file = "evaluation_ground_truth.json"
    if not Path(gt_file).exists():
        print(f"Ground truth file not found: {gt_file}")
        return
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Define configurations to test
    configs = [
        {
            'name': 'Balanced',
            'boost': {'title': 3.0, 'cast': 2.0, 'director': 2.0, 'listed_in': 1.5, 'description': 1.0}
        },
        {
            'name': 'Title Heavy', 
            'boost': {'title': 5.0, 'cast': 2.0, 'director': 1.5, 'listed_in': 1.0, 'description': 0.5}
        },
        {
            'name': 'Cast Heavy',
            'boost': {'title': 2.0, 'cast': 4.0, 'director': 2.0, 'listed_in': 1.0, 'description': 0.5}
        },
        {
            'name': 'Optimized',
            'boost': {'title': 4.0, 'cast': 3.5, 'director': 2.5, 'listed_in': 2.0, 'description': 1.5}
        }
    ]
    
    search_system = ContentSearchSystem(backend_type="minsearch")
    search_system.index_data(csv_path="data/netflix_titles_cleaned.csv")
    evaluator = SearchEvaluator(search_system)
    
    # Test each configuration
    config_objects = []
    config_names = []
    
    for config in configs:
        search_config = SearchConfig(
            boost_weights=config['boost'],
            max_results=50
        )
        config_objects.append({'config': search_config})
        config_names.append(config['name'])
    
    # Run comparison
    comparison_results = evaluator.compare_configurations(
        queries=ground_truth,
        configs=config_objects,
        config_names=config_names
    )
    
    # Display results
    print(f"\n{'='*60}")
    print("CONFIGURATION COMPARISON RESULTS")
    print(f"{'='*60}")
    
    summary = comparison_results['comparison_summary']
    best_config = comparison_results['best_config']
    
    print(f"\n{'Config':<15} {'HR@1':<8} {'HR@5':<8} {'HR@10':<8} {'MRR@10':<8}")
    print("-" * 60)
    
    for config_name in config_names:
        hr1 = summary['hit_rate_at_1'][config_name]
        hr5 = summary['hit_rate_at_5'][config_name]
        hr10 = summary['hit_rate_at_10'][config_name]
        mrr = summary['mrr_at_10'][config_name]
        
        marker = "*" if config_name == best_config else " "
        print(f"{marker} {config_name:<13} {hr1:<8.4f} {hr5:<8.4f} {hr10:<8.4f} {mrr:<8.4f}")
    
    print(f"\nWINNER: {best_config}")
    
    with open("quick_comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"Results saved to: quick_comparison_results.json")
    
    return comparison_results


if __name__ == "__main__":
    quick_comparison()