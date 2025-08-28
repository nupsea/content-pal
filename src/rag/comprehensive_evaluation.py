#!/usr/bin/env python3
"""
Comprehensive evaluation using the full generated ground truth dataset.
"""

import json
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List
import minsearch
import random
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def load_comprehensive_data():
    """Load comprehensive dataset and ground truth"""
    
    print("Loading Netflix dataset...")
    df = pd.read_csv('data/netflix_titles_cleaned.csv', encoding='latin-1')
    df = df.fillna('')
    
    print("Loading comprehensive ground truth...")
    with open('evaluation_ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
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
    
    print(f"Loaded {len(documents)} documents and {len(ground_truth)} ground truth assets")
    return documents, ground_truth

def comprehensive_evaluation():
    """Run comprehensive evaluation with multiple configurations"""
    
    documents, ground_truth = load_comprehensive_data()
    
    # Create minsearch index
    print("Creating minsearch index...")
    index = minsearch.Index(
        text_fields=[
            'title', 'description', 'cast', 'director', 
            'listed_in', 'country', 'rating'
        ],
        keyword_fields=['id', 'type', 'release_year']
    )
    index.fit(documents)
    
    # Test multiple configurations
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
            'name': 'Description Heavy',
            'boost': {'title': 3.0, 'cast': 1.5, 'director': 1.0, 'listed_in': 1.5, 'description': 3.0}
        },
        {
            'name': 'Optimized',
            'boost': {'title': 4.0, 'cast': 3.5, 'director': 2.5, 'listed_in': 2.0, 'description': 1.5}
        }
    ]
    
    print(f"\nTesting {len(configs)} configurations on comprehensive ground truth...")
    print("=" * 80)
    
    best_config = defaultdict()
    best_mrr = 0.0
    all_results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 50)
        
        # Evaluate this configuration
        results = evaluate_config(index, ground_truth, config)
        all_results[config['name']] = results
        
        print(f"Results for {config['name']}:")
        print(f"  Hit Rate@1:  {results['hit_rate_1']:.4f}")
        print(f"  Hit Rate@5:  {results['hit_rate_5']:.4f}")
        print(f"  Hit Rate@10: {results['hit_rate_10']:.4f}")
        print(f"  MRR@10:      {results['mrr_10']:.4f}")
        print(f"  Total queries: {results['total_queries']:,}")
        
        if results['mrr_10'] > best_mrr:
            best_mrr = results['mrr_10']
            best_config = config
    
    # Summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Best Configuration: {best_config['name']}")
    print(f"Best MRR@10: {best_mrr:.4f}")
    
    # Comparison table
    print(f"\nConfiguration Comparison:")
    print(f"{'Config':<15} {'HR@1':<8} {'HR@5':<8} {'HR@10':<8} {'MRR@10':<8}")
    print("-" * 55)
    
    for name, results in all_results.items():
        print(f"{name:<15} {results['hit_rate_1']:<8.4f} {results['hit_rate_5']:<8.4f} "
              f"{results['hit_rate_10']:<8.4f} {results['mrr_10']:<8.4f}")
    
    # Show example searches
    print(f"\nExample searches with best configuration ({best_config['name']}):")
    print("-" * 60)
    
    show_example_searches(index, best_config, ground_truth)
    
    return best_config, all_results

def evaluate_config(index, ground_truth, config, sample_size=3000):
    """Evaluate a single configuration with sampling for efficiency"""
    
    # Sample queries for efficiency
    all_query_pairs = []
    for show_id, queries in ground_truth.items():
        for query in queries:
            all_query_pairs.append((show_id, query))
    
    # Sample if dataset is too large
    if len(all_query_pairs) > sample_size:
        sampled_pairs = random.sample(all_query_pairs, sample_size)
        print(f"  Sampling {sample_size} queries from {len(all_query_pairs)} total")
    else:
        sampled_pairs = all_query_pairs
    
    total_queries = len(sampled_pairs)
    hit_at_1 = 0
    hit_at_5 = 0  
    hit_at_10 = 0
    mrr_sum = 0.0
    
    # Process sampled queries
    for i, (show_id, query) in enumerate(sampled_pairs):
        if i % 500 == 0:
            print(f"    Processed {i}/{total_queries} queries...")
            
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
    
    return {
        'total_queries': total_queries,
        'hit_rate_1': hit_at_1 / total_queries,
        'hit_rate_5': hit_at_5 / total_queries,
        'hit_rate_10': hit_at_10 / total_queries,
        'mrr_10': mrr_sum / total_queries
    }

def show_example_searches(index, config, ground_truth):
    """Show example searches with the best configuration"""
    
    # Select some interesting queries
    sample_queries = [
        ("the matrix", "Popular movie title"),
        ("will smith movies", "Actor search"),
        ("romantic comedies", "Genre search"),
        ("korean shows", "Country/language search"),
        ("horror movies 2020", "Genre + year"),
        ("netflix documentaries", "Platform + genre"),
        ("comedy series", "Genre + type"),
        ("action movies", "Genre search")
    ]
    
    for query, description in sample_queries:
        print(f"\nQuery: '{query}' ({description})")
        
        results = index.search(
            query=query,
            boost_dict=config['boost'], 
            num_results=5
        )
        
        if results:
            print("Top results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['title']} ({result['type']}, {result['release_year']})")
        else:
            print("  No results found")

def analyze_failure_cases(index, ground_truth, config):
    """Analyze queries that return no results"""
    
    print(f"\nAnalyzing failure cases...")
    
    failed_queries = []
    total_queries = 0
    
    for show_id, queries in ground_truth.items():
        for query in queries:
            total_queries += 1
            
            results = index.search(
                query=query,
                boost_dict=config['boost'],
                num_results=10
            )
            
            result_ids = [r['id'] for r in results]
            
            if show_id not in result_ids:
                failed_queries.append((show_id, query, results[:3]))
    
    failure_rate = len(failed_queries) / total_queries
    print(f"Failed queries: {len(failed_queries):,} / {total_queries:,} ({failure_rate:.1%})")
    
    # Show sample failures
    print(f"\nSample failed queries:")
    sample_failures = random.sample(failed_queries, min(10, len(failed_queries)))
    
    for show_id, query, top_results in sample_failures:
        print(f"\nQuery: '{query}' (expected: {show_id})")
        print("Top results instead:")
        for i, result in enumerate(top_results, 1):
            print(f"  {i}. {result['title']} ({result['id']})")

def main():
    """Main evaluation"""
    
    print("Netflix Content Search - Comprehensive Evaluation")
    print("=" * 60)
    
    # Run comprehensive evaluation
    best_config, all_results = comprehensive_evaluation()
    
    # Additional analysis
    documents, ground_truth = load_comprehensive_data()
    index = minsearch.Index(
        text_fields=['title', 'description', 'cast', 'director', 'listed_in', 'country', 'rating'],
        keyword_fields=['id', 'type', 'release_year']
    )
    index.fit(documents)
    
    print(f"\nRunning failure analysis...")
    analyze_failure_cases(index, ground_truth, best_config)
    
    print(f"\nFINAL RESULTS:")
    print(f"Best Configuration: {best_config['name']}")
    print(f"Hit Rate@10: {all_results[best_config['name']]['hit_rate_10']:.4f}")
    print(f"MRR@10: {all_results[best_config['name']]['mrr_10']:.4f}")
    print(f"Evaluated on {all_results[best_config['name']]['total_queries']:,} queries")

if __name__ == "__main__":
    main()