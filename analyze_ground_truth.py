#!/usr/bin/env python3
"""
Analyze the ground truth data quality and create a realistic evaluation dataset.
"""

import json
import pandas as pd
from pathlib import Path

def load_and_analyze():
    # Load data
    df = pd.read_csv('data/netflix_titles.csv', encoding='latin-1')
    with open('notebooks/ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    print("Ground Truth Analysis")
    print("=" * 50)
    
    # Check sample ground truth entries
    sample_ids = ['s1426', 's1047', 's6779', 's5542']
    
    for show_id in sample_ids:
        print(f"\nShow ID: {show_id}")
        queries = ground_truth.get(show_id, [])
        print(f"Ground truth queries: {queries[:3]}...")
        
        # Find actual content
        actual = df[df['show_id'] == show_id]
        if not actual.empty:
            row = actual.iloc[0]
            print(f"Actual title: {row['title']}")
            print(f"Actual description: {row['description'][:100]}...")
            print(f"Listed in: {row['listed_in']}")
            print(f"Cast: {row['cast'][:50]}..." if pd.notna(row['cast']) else "Cast: None")
        else:
            print("❌ Content not found in dataset!")
    
    # Analyze query-content alignment
    print(f"\n{'='*50}")
    print("ALIGNMENT ANALYSIS")
    print("=" * 50)
    
    total_queries = 0
    total_shows = 0
    missing_shows = 0
    
    for show_id, queries in ground_truth.items():
        total_shows += 1
        total_queries += len(queries)
        
        # Check if show exists
        actual = df[df['show_id'] == show_id]
        if actual.empty:
            missing_shows += 1
            continue
            
        row = actual.iloc[0]
        title = str(row['title']).lower()
        desc = str(row['description']).lower() if pd.notna(row['description']) else ""
        cast = str(row['cast']).lower() if pd.notna(row['cast']) else ""
        genres = str(row['listed_in']).lower() if pd.notna(row['listed_in']) else ""
        
        # Check some queries for alignment
        for query in queries[:2]:  # Check first 2 queries
            query_lower = query.lower()
            
            # Simple alignment checks
            title_match = any(word in title for word in query_lower.split()[:3])
            desc_match = any(word in desc for word in query_lower.split()[:3])
            cast_match = any(word in cast for word in query_lower.split()[:3])
            genre_match = any(word in genres for word in query_lower.split()[:3])
            
            if not (title_match or desc_match or cast_match or genre_match):
                print(f"❌ Poor alignment - {show_id}: '{query}' vs '{row['title']}'")
    
    print(f"\nSUMMARY:")
    print(f"Total shows in ground truth: {total_shows}")
    print(f"Missing shows: {missing_shows} ({missing_shows/total_shows*100:.1f}%)")
    print(f"Total queries: {total_queries}")
    print(f"Average queries per show: {total_queries/total_shows:.1f}")
    
    # Check for realistic content
    print(f"\n{'='*50}")
    print("CONTENT CATEGORIES")
    print("=" * 50)
    
    # Check popular actors
    popular_actors = ['will smith', 'leonardo dicaprio', 'brad pitt', 'jennifer lawrence', 'ryan reynolds']
    for actor in popular_actors:
        matches = df[df['cast'].str.contains(actor, case=False, na=False)]
        print(f"{actor.title()}: {len(matches)} movies/shows")
    
    # Check popular genres  
    print("\nGenre distribution:")
    all_genres = []
    for genres in df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    
    from collections import Counter
    genre_counts = Counter(all_genres)
    for genre, count in genre_counts.most_common(10):
        print(f"{genre}: {count}")

if __name__ == "__main__":
    load_and_analyze()