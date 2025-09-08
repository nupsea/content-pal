#!/usr/bin/env python3
"""
Create Proper Enriched Dataset for Evaluation

The issue: 10-movie sample doesn't cover evaluation ground truth movies.
Solution: Create comprehensive enriched dataset with 1000+ movies to ensure coverage.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing.no_api_enrichment import NoAPIEnrichment
import json
import pandas as pd

def main():
    """Create proper enriched dataset for evaluation"""
    print("🔧 CREATING PROPER ENRICHED DATASET FOR EVALUATION")
    print("=" * 60)
    
    # Load ground truth to understand what movies we need to cover
    try:
        with open("new_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        gt_movie_ids = set(ground_truth.keys())
        print(f"📋 Ground truth contains: {len(gt_movie_ids)} unique movies")
        
        # Load Netflix dataset to check coverage
        df = pd.read_csv("data/netflix_titles_cleaned.csv", encoding='latin-1')
        netflix_ids = set(df['show_id'].astype(str))
        
        # Find overlap
        overlap = gt_movie_ids.intersection(netflix_ids)
        print(f"🎯 Movies in both GT and Netflix: {len(overlap)}")
        print(f"📊 Coverage needed: {len(overlap)}/{len(gt_movie_ids)} ({100*len(overlap)/len(gt_movie_ids):.1f}%)")
        
    except Exception as e:
        print(f"⚠️  Could not load ground truth: {e}")
        print("📝 Will create large sample for comprehensive coverage")
    
    print("\n🚀 SOLUTION: Create 1500-movie enriched dataset")
    print("This ensures comprehensive coverage for evaluation")
    print("Expected time: ~45 minutes with respectful rate limiting")
    print()
    
    # Confirm before processing
    response = input("Proceed with 1500-movie enrichment? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ Cancelled. Try smaller dataset with:")
        print("   python create_comprehensive_no_api_dataset.py  # 200 movies")
        return
    
    print("\n🔄 Starting comprehensive enrichment...")
    
    # Create enricher
    enricher = NoAPIEnrichment()
    
    # Process substantial dataset to ensure evaluation coverage
    enriched_df = enricher.enrich_dataset_no_api(
        input_csv="data/netflix_titles_cleaned.csv",
        output_csv="data/netflix_comprehensive_enriched.csv",
        max_movies=1500  # Large enough to cover evaluation needs
    )
    
    # Statistics
    print("\n📊 COMPREHENSIVE ENRICHMENT RESULTS:")
    print(f"   Total movies processed: {len(enriched_df)}")
    
    non_empty_tags = enriched_df[enriched_df['no_api_enhanced_tags'] != '']
    success_rate = len(non_empty_tags) / len(enriched_df) * 100
    print(f"   Successfully enriched: {len(non_empty_tags)} ({success_rate:.1f}%)")
    
    if len(non_empty_tags) > 0:
        avg_tags = non_empty_tags['no_api_enhanced_tags'].str.split('|').str.len().mean()
        print(f"   Average tags per movie: {avg_tags:.1f}")
    
    # Show coverage analysis
    try:
        enriched_ids = set(enriched_df['show_id'].astype(str))
        covered_gt_movies = gt_movie_ids.intersection(enriched_ids)
        coverage = len(covered_gt_movies) / len(gt_movie_ids) * 100
        print(f"   Ground truth coverage: {len(covered_gt_movies)}/{len(gt_movie_ids)} ({coverage:.1f}%)")
    except:
        pass
    
    # Sample results
    print("\n🎯 Sample enriched movies:")
    for i in range(min(5, len(enriched_df))):
        row = enriched_df.iloc[i]
        tags = row['no_api_enhanced_tags']
        print(f"\n{i+1}. {row['title']} ({row['release_year']})")
        if tags:
            tag_list = tags.split('|')[:8]
            print(f"   Tags: {' | '.join(tag_list)}...")
        else:
            print("   Tags: (none found)")
    
    print(f"\n✅ COMPREHENSIVE DATASET READY!")
    print(f"📁 Saved to: data/netflix_comprehensive_enriched.csv")
    print(f"🧪 Ready for proper evaluation testing!")
    
    print("\n🚀 Next steps:")
    print("1. Update evaluation to use comprehensive dataset:")
    print("   cp data/netflix_comprehensive_enriched.csv data/netflix_titles_enriched.csv")
    print("2. Run comprehensive evaluation:")
    print("   pipenv run python src/run_comprehensive_evaluation.py")
    print("3. Expected: Significant improvement over 0.0000 results!")


if __name__ == "__main__":
    main()