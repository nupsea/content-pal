#!/usr/bin/env python3
"""
Create Comprehensive No-API Enriched Dataset

Generates a substantial dataset using Wikipedia + web scraping
for better search performance without requiring API keys.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing.no_api_enrichment import NoAPIEnrichment

def main():
    """Create comprehensive no-API enriched dataset"""
    print("🆓 COMPREHENSIVE NO-API DATASET CREATION")
    print("=" * 50)
    print("This will enrich 200 movies using Wikipedia + web scraping")
    print("Expected time: ~20 minutes (respectful rate limiting)")
    print()
    
    # Create enricher
    enricher = NoAPIEnrichment()
    
    print("🔄 Processing 200 movies for comprehensive testing...")
    
    # Process a substantial sample (200 movies)
    enriched_df = enricher.enrich_dataset_no_api(
        input_csv="data/netflix_titles_cleaned.csv",
        output_csv="data/netflix_no_api_enriched_200.csv",
        max_movies=200
    )
    
    # Show comprehensive results
    print("\n📊 ENRICHMENT SUMMARY:")
    print(f"   Total movies: {len(enriched_df)}")
    
    non_empty_tags = enriched_df[enriched_df['no_api_enhanced_tags'] != '']
    print(f"   Successfully enriched: {len(non_empty_tags)}")
    
    if len(non_empty_tags) > 0:
        avg_tags = non_empty_tags['no_api_enhanced_tags'].str.split('|').str.len().mean()
        print(f"   Average tags per movie: {avg_tags:.1f}")
    
    print("\n🎯 Sample enriched movies:")
    for i in range(min(5, len(enriched_df))):
        row = enriched_df.iloc[i]
        tags = row['no_api_enhanced_tags']
        print(f"\n{i+1}. {row['title']} ({row['release_year']})")
        if tags:
            tag_list = tags.split('|')
            if len(tag_list) > 8:
                shown_tags = tag_list[:8]
                print(f"   Tags: {' | '.join(shown_tags)}...")
                print(f"   (+{len(tag_list) - 8} more)")
            else:
                print(f"   Tags: {' | '.join(tag_list)}")
        else:
            print("   Tags: (none found)")
    
    print(f"\n✅ Comprehensive no-API dataset ready!")
    print(f"📁 Saved to: data/netflix_no_api_enriched_200.csv")
    print("\n🚀 Next steps:")
    print("1. Test with enriched search system")
    print("2. Run comprehensive evaluation")
    print("3. Compare with API-based approaches")

if __name__ == "__main__":
    main()