#!/usr/bin/env python3
"""
Generate realistic, unbiased ground truth for comprehensive evaluation
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from modules.evaluation.realistic_ground_truth import RealisticGroundTruthGenerator


def generate_and_analyze():
    """Generate realistic ground truth and analyze it"""
    
    print("🎯 Generating Realistic Ground Truth")
    print("=" * 50)
    
    generator = RealisticGroundTruthGenerator()
    
    # Generate realistic ground truth
    ground_truth = generator.generate_realistic_ground_truth(
        csv_path="data/netflix_titles_cleaned.csv",
        output_path="realistic_ground_truth.json", 
        num_queries=600  # Good balance of coverage and evaluation time
    )
    
    # Analyze the generated ground truth
    print(f"\n📊 Ground Truth Analysis:")
    print(f"Total queries: {len(ground_truth)}")
    
    # Analyze query lengths and complexity
    query_lengths = [len(query.split()) for query in ground_truth.keys()]
    avg_length = sum(query_lengths) / len(query_lengths)
    print(f"Average query length: {avg_length:.1f} words")
    
    # Analyze result set sizes
    result_counts = [len(results) for results in ground_truth.values()]
    avg_results = sum(result_counts) / len(result_counts)
    print(f"Average relevant results per query: {avg_results:.1f}")
    
    # Show sample queries by category
    print(f"\n🔍 Sample Query Categories:")
    
    temporal_queries = [q for q in ground_truth.keys() if any(word in q.lower() for word in ['after', 'recent', '20', '90s', '2000s'])]
    print(f"Temporal queries ({len(temporal_queries)}): ")
    for q in temporal_queries[:3]:
        print(f"  • '{q}'")
    
    mood_queries = [q for q in ground_truth.keys() if any(word in q.lower() for word in ['feel-good', 'dark', 'family', 'funny'])]
    print(f"\nMood-based queries ({len(mood_queries)}):")
    for q in mood_queries[:3]:
        print(f"  • '{q}'")
    
    actor_queries = [q for q in ground_truth.keys() if any(word in q.lower() for word in ['with', 'starring', 'featuring'])]
    print(f"\nActor queries ({len(actor_queries)}):")
    for q in actor_queries[:3]:
        print(f"  • '{q}'")
    
    thematic_queries = [q for q in ground_truth.keys() if 'about' in q.lower()]
    print(f"\nThematic queries ({len(thematic_queries)}):")
    for q in thematic_queries[:3]:
        print(f"  • '{q}'")
    
    # Compare with biased ground truth structure
    print(f"\n🆚 Comparison with Biased Approach:")
    print(f"Realistic approach:")
    print(f"  ✅ Natural language queries")
    print(f"  ✅ Multiple relevant results per query")
    print(f"  ✅ Reflects real user search patterns")
    print(f"  ✅ Tests semantic understanding")
    print(f"  ✅ Includes temporal and mood constraints")
    
    print(f"\nBiased approach:")
    print(f"  ❌ Artificial field-based queries")
    print(f"  ❌ Perfect entity matching")
    print(f"  ❌ Favors keyword search")
    print(f"  ❌ Doesn't test real-world scenarios")
    
    return ground_truth


def create_evaluation_subset(ground_truth: dict, subset_size: int = 200):
    """Create a smaller subset for faster evaluation"""
    
    print(f"\n📝 Creating evaluation subset ({subset_size} queries)...")
    
    # Select diverse subset
    import random
    random.seed(42)  # Reproducible results
    
    queries = list(ground_truth.items())
    random.shuffle(queries)
    subset = dict(queries[:subset_size])
    
    # Save subset
    with open("realistic_evaluation_subset.json", "w") as f:
        json.dump(subset, f, indent=2)
    
    print(f"💾 Saved evaluation subset to realistic_evaluation_subset.json")
    return subset


if __name__ == "__main__":
    # Generate realistic ground truth
    ground_truth = generate_and_analyze()
    
    # Create evaluation subset
    subset = create_evaluation_subset(ground_truth, subset_size=200)
    
    print(f"\n🎉 Realistic Ground Truth Generation Complete!")
    print(f"Ready for unbiased comprehensive evaluation.")
    print(f"\nTo run evaluation:")
    print(f"  pipenv run python src/run_realistic_evaluation.py")