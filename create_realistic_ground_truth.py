#!/usr/bin/env python3
"""
Create a realistic ground truth dataset based on actual Netflix content.
Focus on queries that users would actually search for and can be found.
"""

import json
import pandas as pd
import random
from collections import defaultdict

def create_realistic_ground_truth():
    """Create ground truth based on actual content characteristics"""
    
    df = pd.read_csv('data/netflix_titles.csv', encoding='latin-1')
    
    # Clean the data
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['cast'] = df['cast'].fillna('')
    df['director'] = df['director'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    
    ground_truth = {}
    
    print("Creating realistic ground truth dataset...")
    print("=" * 50)
    
    # Strategy 1: ACTOR-based queries (most reliable)
    print("Creating actor-based queries...")
    actor_queries = 0
    
    for _, row in df.iterrows():
        if not row['cast']:
            continue
            
        # Get first 2 actors (most prominent)
        actors = [actor.strip() for actor in row['cast'].split(',')[:2]]
        
        queries = []
        for actor in actors:
            if len(actor) > 3 and ' ' in actor:  # Full names only
                queries.extend([
                    f"movies with {actor}",
                    f"{actor} films",
                    f"shows starring {actor}"
                ])
        
        if queries:
            ground_truth[row['show_id']] = queries[:3]  # Max 3 per show
            actor_queries += 1
            
        if actor_queries >= 150:  # Limit to prevent too many
            break
    
    # Strategy 2: TITLE-based queries
    print("Creating title-based queries...")
    title_queries = 0
    
    for _, row in df.iterrows():
        title = row['title'].strip()
        if len(title) > 2 and title not in ['', 'N/A']:
            # Create partial title queries
            words = title.split()
            if len(words) >= 2:
                queries = [
                    title,  # Exact title
                    ' '.join(words[:2]),  # First 2 words
                ]
                
                # Add with movie/show qualifier
                content_type = "movie" if row['type'] == 'Movie' else "show"
                queries.append(f"{words[0]} {content_type}")
                
                ground_truth[row['show_id']] = queries[:3]
                title_queries += 1
                
            if title_queries >= 100:
                break
    
    # Strategy 3: GENRE + YEAR queries (realistic combinations)
    print("Creating genre+year queries...")
    genre_year_queries = 0
    
    # Focus on popular genres and recent years
    popular_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller', 'Documentary']
    recent_years = [2019, 2020, 2021, 2022, 2023]
    
    for _, row in df.iterrows():
        if pd.isna(row['release_year']) or row['release_year'] not in recent_years:
            continue
            
        genres = [g.strip() for g in row['listed_in'].split(',')]
        matching_genres = [g for g in genres if any(pg in g for pg in popular_genres)]
        
        if matching_genres:
            year = int(row['release_year'])
            genre = matching_genres[0]
            
            queries = [
                f"{genre.lower()} movies {year}",
                f"{year} {genre.lower()} films",
                f"best {genre.lower()} {year}"
            ]
            
            ground_truth[row['show_id']] = queries[:3]
            genre_year_queries += 1
            
        if genre_year_queries >= 50:
            break
    
    # Strategy 4: DIRECTOR-based queries (for well-known directors)
    print("Creating director-based queries...")
    director_queries = 0
    
    famous_directors = [
        'Martin Scorsese', 'Christopher Nolan', 'Quentin Tarantino', 
        'Steven Spielberg', 'David Fincher', 'Ridley Scott',
        'Tim Burton', 'Guillermo del Toro', 'Jordan Peele'
    ]
    
    for _, row in df.iterrows():
        director = row['director'].strip()
        if director and any(fd in director for fd in famous_directors):
            queries = [
                f"{director} movies",
                f"films directed by {director}",
                f"{director} latest film"
            ]
            
            ground_truth[row['show_id']] = queries[:3]
            director_queries += 1
            
        if director_queries >= 30:
            break
    
    # Strategy 5: CONTENT DESCRIPTION queries (semantic)
    print("Creating description-based queries...")
    desc_queries = 0
    
    # Look for distinctive plot elements
    plot_keywords = [
        'zombie', 'vampire', 'detective', 'murder', 'heist', 'robot', 
        'alien', 'superhero', 'spy', 'high school', 'college', 'family'
    ]
    
    for _, row in df.iterrows():
        desc = row['description'].lower()
        
        for keyword in plot_keywords:
            if keyword in desc:
                queries = [
                    f"{keyword} movies",
                    f"films about {keyword}",
                    f"{keyword} stories"
                ]
                
                ground_truth[row['show_id']] = queries[:3]
                desc_queries += 1
                break
                
        if desc_queries >= 50:
            break
    
    # Remove duplicates and validate
    print(f"\nValidating ground truth...")
    final_ground_truth = {}
    
    for show_id, queries in ground_truth.items():
        # Remove duplicates and empty queries
        unique_queries = list(set([q for q in queries if q and len(q) > 3]))
        
        if len(unique_queries) >= 2:  # At least 2 good queries
            final_ground_truth[show_id] = unique_queries[:5]  # Max 5 per show
    
    print(f"SUMMARY:")
    print(f"Actor-based queries: {actor_queries}")
    print(f"Title-based queries: {title_queries}")  
    print(f"Genre+year queries: {genre_year_queries}")
    print(f"Director-based queries: {director_queries}")
    print(f"Description-based queries: {desc_queries}")
    print(f"Total shows: {len(final_ground_truth)}")
    print(f"Total queries: {sum(len(qs) for qs in final_ground_truth.values())}")
    
    # Save to file
    output_file = 'realistic_ground_truth.json'
    with open(output_file, 'w') as f:
        json.dump(final_ground_truth, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    
    # Show some examples
    print(f"\nEXAMPLES:")
    sample_ids = list(final_ground_truth.keys())[:10]
    for show_id in sample_ids:
        # Get actual content info
        content = df[df['show_id'] == show_id].iloc[0]
        print(f"\n{show_id}: {content['title']}")
        print(f"Queries: {final_ground_truth[show_id]}")
    
    return final_ground_truth

if __name__ == "__main__":
    create_realistic_ground_truth()