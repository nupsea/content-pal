#!/usr/bin/env python3
"""
Validate and analyze the comprehensive ground truth dataset.
"""

import json
import pandas as pd
import random
from collections import Counter, defaultdict

def analyze_ground_truth():
    """Analyze the comprehensive ground truth dataset"""
    
    print("Loading comprehensive ground truth...")
    with open('comprehensive_ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    print("Loading Netflix dataset...")
    df = pd.read_csv('data/netflix_titles_cleaned.csv', encoding='latin-1')
    
    print("=" * 60)
    print("COMPREHENSIVE GROUND TRUTH ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    total_assets = len(ground_truth)
    total_queries = sum(len(queries) for queries in ground_truth.values())
    avg_queries = total_queries / total_assets
    
    print(f"\nBASIC STATISTICS:")
    print(f"Total assets with ground truth: {total_assets:,}")
    print(f"Total queries generated: {total_queries:,}")
    print(f"Average queries per asset: {avg_queries:.1f}")
    
    # Query length analysis
    all_queries = []
    for queries in ground_truth.values():
        all_queries.extend(queries)
    
    query_lengths = [len(q.split()) for q in all_queries]
    query_chars = [len(q) for q in all_queries]
    
    print(f"\nQUERY CHARACTERISTICS:")
    print(f"Average query length (words): {sum(query_lengths) / len(query_lengths):.1f}")
    print(f"Average query length (chars): {sum(query_chars) / len(query_chars):.1f}")
    print(f"Shortest query: '{min(all_queries, key=len)}'")
    print(f"Longest query: '{max(all_queries, key=len)}'")
    
    # Query type analysis
    print(f"\nQUERY TYPE DISTRIBUTION:")
    
    # Analyze query patterns
    title_queries = sum(1 for q in all_queries if any(word in q for word in ['watch', 'netflix']))
    actor_queries = sum(1 for q in all_queries if 'movies' in q or 'shows' in q or 'films' in q)
    genre_queries = sum(1 for q in all_queries if any(word in q for word in ['comedy', 'drama', 'action', 'horror', 'thriller']))
    year_queries = sum(1 for q in all_queries if any(str(year) in q for year in range(1990, 2025)))
    
    print(f"Title/Platform queries (~): {title_queries:,} ({title_queries/total_queries*100:.1f}%)")
    print(f"Actor/Content queries (~): {actor_queries:,} ({actor_queries/total_queries*100:.1f}%)")
    print(f"Genre queries (~): {genre_queries:,} ({genre_queries/total_queries*100:.1f}%)")
    print(f"Year queries (~): {year_queries:,} ({year_queries/total_queries*100:.1f}%)")
    
    # Content type breakdown
    print(f"\nCONTENT TYPE BREAKDOWN:")
    movie_assets = sum(1 for show_id in ground_truth.keys() 
                      if show_id in df['show_id'].values and df[df['show_id'] == show_id]['type'].iloc[0] == 'Movie')
    tv_assets = total_assets - movie_assets
    
    print(f"Movies: {movie_assets:,} ({movie_assets/total_assets*100:.1f}%)")
    print(f"TV Shows: {tv_assets:,} ({tv_assets/total_assets*100:.1f}%)")
    
    # Show examples by category
    print(f"\nSAMPLE QUERIES BY CATEGORY:")
    print("=" * 40)
    
    # Get sample assets
    sample_assets = random.sample(list(ground_truth.keys()), 10)
    
    for show_id in sample_assets:
        content = df[df['show_id'] == show_id]
        if not content.empty:
            row = content.iloc[0]
            queries = ground_truth[show_id]
            
            print(f"\n{show_id}: {row['title']} ({row['type']}, {row['release_year']})")
            print(f"Genres: {row['listed_in']}")
            print(f"Sample queries ({len(queries)} total):")
            for i, query in enumerate(queries[:5], 1):
                print(f"  {i}. '{query}'")
    
    # Query diversity analysis
    print(f"\n{'='*60}")
    print("QUERY DIVERSITY ANALYSIS")
    print("=" * 60)
    
    # Most common query terms
    all_words = []
    for query in all_queries:
        all_words.extend(query.split())
    
    word_counts = Counter(all_words)
    print(f"\nMost common query terms:")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count:,} ({count/len(all_words)*100:.2f}%)")
    
    # Check for duplicates
    query_counts = Counter(all_queries)
    duplicates = {q: count for q, count in query_counts.items() if count > 1}
    
    print(f"\nDuplicate queries: {len(duplicates)} unique queries appear multiple times")
    if duplicates:
        print("Most frequent duplicates:")
        for query, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  '{query}': {count} times")
    
    # Validate content alignment
    print(f"\n{'='*60}")
    print("CONTENT ALIGNMENT VALIDATION")
    print("=" * 60)
    
    # Check random samples for alignment
    validation_samples = random.sample(list(ground_truth.keys()), 20)
    alignment_issues = 0
    
    print("Validating query-content alignment...")
    for show_id in validation_samples:
        content = df[df['show_id'] == show_id]
        if content.empty:
            continue
            
        row = content.iloc[0]
        queries = ground_truth[show_id]
        
        title = str(row['title']).lower()
        cast = str(row['cast']).lower() if pd.notna(row['cast']) else ""
        director = str(row['director']).lower() if pd.notna(row['director']) else ""
        
        # Check if queries align with content
        title_matches = sum(1 for q in queries if any(word in title for word in q.split()[:2]))
        cast_matches = sum(1 for q in queries if any(word in cast for word in q.split()[:2]))
        
        total_matches = title_matches + cast_matches
        alignment_score = total_matches / len(queries)
        
        if alignment_score < 0.3:  # Less than 30% alignment
            alignment_issues += 1
            print(f"⚠️  {show_id}: {row['title']} - Low alignment ({alignment_score:.2f})")
    
    print(f"Alignment issues: {alignment_issues}/{len(validation_samples)} samples")
    
    return ground_truth

def create_evaluation_subset():
    """Create a manageable subset for evaluation"""
    
    with open('comprehensive_ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    print(f"\n{'='*60}")
    print("CREATING EVALUATION SUBSET")
    print("=" * 60)
    
    # Create balanced subset for evaluation
    subset_size = 1000  # Manageable size for evaluation
    
    # Randomly sample assets
    asset_ids = list(ground_truth.keys())
    selected_assets = random.sample(asset_ids, min(subset_size, len(asset_ids)))
    
    eval_ground_truth = {asset_id: ground_truth[asset_id] for asset_id in selected_assets}
    
    # Save evaluation subset
    with open('evaluation_ground_truth.json', 'w') as f:
        json.dump(eval_ground_truth, f, indent=2)
    
    total_eval_queries = sum(len(queries) for queries in eval_ground_truth.values())
    
    print(f"Created evaluation subset:")
    print(f"Assets: {len(eval_ground_truth):,}")
    print(f"Queries: {total_eval_queries:,}")
    print(f"Average queries per asset: {total_eval_queries/len(eval_ground_truth):.1f}")
    print(f"Saved to: evaluation_ground_truth.json")
    
    return eval_ground_truth

def main():
    # Analyze the comprehensive ground truth
    ground_truth = analyze_ground_truth()
    
    # Create evaluation subset
    eval_subset = create_evaluation_subset()
    
    print(f"\n🎉 Validation complete!")
    print(f"Comprehensive dataset: {len(ground_truth):,} assets, {sum(len(q) for q in ground_truth.values()):,} queries")
    print(f"Evaluation subset: {len(eval_subset):,} assets, {sum(len(q) for q in eval_subset.values()):,} queries")

if __name__ == "__main__":
    main()