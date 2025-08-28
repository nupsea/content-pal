#!/usr/bin/env python3
"""
Simple evaluation using minsearch - lightweight, effective search.
"""

import json
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List
import minsearch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def prepare_documents():
    """Load and prepare Netflix documents for minsearch"""
    
    print("Loading Netflix data...")
    df = pd.read_csv('data/netflix_titles.csv', encoding='latin-1')
    
    # Clean data
    df = df.fillna('')
    
    # Prepare documents for minsearch
    documents = []
    for _, row in df.iterrows():
        doc = {
            'id': row['show_id'],
            'title': row['title'],
            'description': row['description'], 
            'cast': row['cast'],
            'director': row['director'],
            'listed_in': row['listed_in'],
            'type': row['type'],
            'release_year': str(row['release_year']) if pd.notna(row['release_year']) else '',
            'country': row['country'],
            'rating': row['rating'],
            'duration': row['duration']
        }
        documents.append(doc)
    
    print(f"Prepared {len(documents)} documents")
    return documents

def create_search_index(documents):
    """Create minsearch index"""
    
    print("Creating minsearch index...")
    
    # Create index with important fields
    index = minsearch.Index(
        text_fields=[
            'title', 'description', 'cast', 'director', 
            'listed_in', 'country', 'rating'
        ],
        keyword_fields=['id', 'type', 'release_year']
    )
    
    # Fit the index
    index.fit(documents)
    
    print("Index created successfully")
    return index

def evaluate_search(index, ground_truth_file: str):
    """Evaluate search performance"""
    
    print(f"Loading ground truth from {ground_truth_file}...")
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"Loaded {len(ground_truth)} shows with ground truth queries")
    
    # Evaluation metrics
    total_queries = 0
    hit_at_1 = 0
    hit_at_5 = 0  
    hit_at_10 = 0
    mrr_sum = 0.0
    
    # Test different search configurations
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
        }
    ]
    
    best_config = None
    best_mrr = 0.0
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        print("-" * 40)
        
        total_queries = 0
        hit_at_1 = 0
        hit_at_5 = 0
        hit_at_10 = 0
        mrr_sum = 0.0
        
        for show_id, queries in ground_truth.items():
            for query in queries:
                total_queries += 1
                
                # Search with current config
                results = index.search(
                    query=query,
                    boost_dict=config['boost'],
                    num_results=10
                )
                
                # Extract result IDs
                result_ids = [r['id'] for r in results]
                
                # Calculate metrics
                if show_id in result_ids:
                    rank = result_ids.index(show_id) + 1
                    
                    if rank == 1:
                        hit_at_1 += 1
                    if rank <= 5:
                        hit_at_5 += 1
                    if rank <= 10:
                        hit_at_10 += 1
                    
                    # MRR contribution
                    mrr_sum += 1.0 / rank
        
        # Calculate final metrics
        hit_rate_1 = hit_at_1 / total_queries
        hit_rate_5 = hit_at_5 / total_queries
        hit_rate_10 = hit_at_10 / total_queries
        mrr = mrr_sum / total_queries
        
        print(f"Results for {config['name']}:")
        print(f"  Total queries: {total_queries}")
        print(f"  Hit Rate@1:  {hit_rate_1:.4f} ({hit_at_1}/{total_queries})")
        print(f"  Hit Rate@5:  {hit_rate_5:.4f} ({hit_at_5}/{total_queries})")
        print(f"  Hit Rate@10: {hit_rate_10:.4f} ({hit_at_10}/{total_queries})")
        print(f"  MRR@10:      {mrr:.4f}")
        
        if mrr > best_mrr:
            best_mrr = mrr
            best_config = config
    
    print(f"\n{'='*50}")
    print(f"BEST CONFIGURATION: {best_config['name']}")
    print(f"Best MRR@10: {best_mrr:.4f}")
    print(f"{'='*50}")
    
    # Show some example searches with best config
    print(f"\nExample searches with best configuration:")
    print("-" * 40)
    
    sample_queries = [
        ("Will Smith movies", "s-will-smith"),
        ("The Matrix", "s-matrix"), 
        ("romantic comedies", "s-romance"),
        ("vampire movies", "s-vampire")
    ]
    
    for query, label in sample_queries:
        print(f"\nQuery: '{query}'")
        results = index.search(
            query=query,
            boost_dict=best_config['boost'], 
            num_results=5
        )
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} ({result['type']}, {result['release_year']})")
    
    return best_config, best_mrr

def main():
    """Main evaluation pipeline"""
    
    print("Netflix Search Evaluation with MinSearch")
    print("=" * 50)
    
    # Prepare data
    documents = prepare_documents()
    
    # Create search index
    index = create_search_index(documents)
    
    # Evaluate with realistic ground truth
    best_config, best_mrr = evaluate_search(index, 'realistic_ground_truth.json')
    
    print(f"\nFINAL RESULTS:")
    print(f"Best MRR@10: {best_mrr:.4f}")
    print(f"Using configuration: {best_config['name']}")
    
    # Compare with original ground truth (if available)
    try:
        print(f"\n{'='*50}")
        print("COMPARISON WITH ORIGINAL GROUND TRUTH")
        print("=" * 50)
        
        original_config, original_mrr = evaluate_search(index, 'notebooks/ground_truth.json')
        
        print(f"\nCOMPARISON:")
        print(f"Original ground truth MRR: {original_mrr:.4f}")
        print(f"Realistic ground truth MRR: {best_mrr:.4f}")
        print(f"Improvement: {((best_mrr - original_mrr) / original_mrr * 100):.1f}%")
        
    except FileNotFoundError:
        print("Original ground truth not found - skipping comparison")

if __name__ == "__main__":
    main()