#!/usr/bin/env python3
"""
Generate full enriched dataset with model-based tags
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing.model_based_tagger import ModelBasedTagger

def main():
    """Generate full enriched dataset"""
    print("🚀 GENERATING FULL ENRICHED DATASET")
    print("=" * 60)
    print("This will process all 7,370 Netflix movies with model-based tagging...")
    print("Estimated time: 15-20 minutes")
    
    # Auto-proceed for non-interactive execution
    print("\n🔄 Starting automatic enrichment...")
    
    # Create tagger and process full dataset
    tagger = ModelBasedTagger()
    
    input_file = "data/netflix_titles_cleaned.csv"
    output_file = "data/netflix_titles_enriched.csv"
    
    print(f"\n🔄 Processing full dataset...")
    enriched_df = tagger.enrich_dataset(input_file, output_file)
    
    print(f"\n✅ Full enriched dataset generated!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Processed: {len(enriched_df)} movies")
    
    # Show sample results
    print("\n🎯 Sample enriched movies:")
    for i in range(3):
        if i < len(enriched_df):
            row = enriched_df.iloc[i]
            print(f"\n{i+1}. {row['title']} ({row['release_year']})")
            tags = row['semantic_tags']
            if tags:
                tag_list = tags.split('|')[:8]
                print(f"   Tags: {' | '.join(tag_list)}{'...' if len(tags.split('|')) > 8 else ''}")

if __name__ == "__main__":
    main()