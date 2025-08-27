"""
Ground truth generation utilities
"""

import json
import pandas as pd
import re
import random
from typing import List, Dict, Set
from collections import defaultdict


class GroundTruthGenerator:
    """Generates realistic ground truth queries for evaluation"""
    
    def __init__(self):
        self.genre_mappings = {
            'Action & Adventure': ['action movies', 'adventure films'],
            'Comedies': ['comedy movies', 'funny movies', 'comedies'],
            'Documentaries': ['documentaries', 'documentary films'],
            'Dramas': ['drama movies', 'drama films', 'dramas'],
            'Horror': ['horror movies', 'scary movies'],
            'Romantic Movies': ['romantic movies', 'romance films'],
            'Thrillers': ['thriller movies', 'thriller films'],
            'Sci-Fi & Fantasy': ['sci-fi movies', 'fantasy films'],
            'Crime': ['crime movies', 'crime shows'],
            'TV Dramas': ['tv dramas', 'drama series'],
            'TV Comedies': ['tv comedies', 'comedy series'],
            'International Movies': ['foreign movies', 'international films'],
            'Korean TV Shows': ['k-dramas', 'korean series'],
            'British TV Shows': ['british shows', 'uk series'],
            'Children & Family Movies': ['kids movies', 'family films'],
            'Anime': ['anime series', 'anime shows']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or text == '':
            return ''
        return str(text).strip()
    
    def extract_main_actors(self, cast_str: str, max_actors: int = 3) -> List[str]:
        """Extract main actors from cast string"""
        cast = self.clean_text(cast_str)
        if not cast:
            return []
        
        actors = []
        for actor in cast.split(',')[:max_actors]:
            actor = actor.strip()
            # Basic validation for actor names
            if len(actor) > 5 and ' ' in actor and actor.replace(' ', '').replace("'", "").replace("-", "").isalpha():
                actors.append(actor)
        return actors
    
    def extract_genres(self, listed_in: str) -> List[str]:
        """Extract genres from listed_in field"""
        genres_str = self.clean_text(listed_in)
        if not genres_str:
            return []
        
        genres = []
        for genre in genres_str.split(','):
            genre = genre.strip()
            if genre and len(genre) > 2:
                genres.append(genre)
        return genres
    
    def generate_title_queries(self, title: str) -> List[str]:
        """Generate title-based queries"""
        if not title:
            return []
        
        queries = [title]  # Exact title
        
        # Add variations
        words = title.split()
        if len(words) >= 2:
            queries.append(' '.join(words[:2]))  # First two words
            if len(words) >= 3:
                queries.append(' '.join(words[:3]))  # First three words
        
        # Add with qualifiers
        queries.extend([
            f"{title} movie",
            f"{title} show",
            f"{title} netflix",
            f"watch {title}"
        ])
        
        return queries
    
    def generate_actor_queries(self, actors: List[str], content_type: str) -> List[str]:
        """Generate actor-based queries"""
        queries = []
        media_type = 'movies' if content_type == 'Movie' else 'shows'
        
        for actor in actors:
            queries.extend([
                f"{actor} {media_type}",
                f"movies with {actor}",
                f"shows with {actor}",
                f"{actor} films",
                f"{actor} netflix"
            ])
        
        return queries
    
    def generate_genre_queries(self, genres: List[str], release_year: int = None) -> List[str]:
        """Generate genre-based queries"""
        queries = []
        
        for genre in genres[:2]:  # Use top 2 genres
            # Map to natural query terms
            natural_queries = self.genre_mappings.get(genre, [genre.lower()])
            
            for natural_genre in natural_queries:
                queries.extend([
                    natural_genre,
                    f"best {natural_genre}",
                    f"new {natural_genre}",
                    f"{natural_genre} netflix"
                ])
                
                # Add year-based queries if year available
                if release_year:
                    queries.append(f"{natural_genre} {release_year}")
                    if 2020 <= release_year <= 2024:
                        queries.append(f"{natural_genre} 2020s")
        
        return queries
    
    def generate_director_queries(self, director: str, content_type: str) -> List[str]:
        """Generate director-based queries"""
        director = self.clean_text(director)
        if not director or len(director) < 5:
            return []
        
        media_type = 'movies' if content_type == 'Movie' else 'shows'
        
        return [
            f"{director} {media_type}",
            f"films by {director}",
            f"{director} director",
            f"{director} netflix"
        ]
    
    def generate_year_queries(self, release_year: int, content_type: str) -> List[str]:
        """Generate year-based queries"""
        if not release_year:
            return []
        
        media_type = 'movies' if content_type == 'Movie' else 'shows'
        
        queries = [
            f"{release_year} {media_type}",
            f"best {media_type} {release_year}"
        ]
        
        # Add recent qualifiers for recent content
        if release_year >= 2020:
            queries.extend([
                f"new {media_type}",
                f"recent {media_type}",
                f"latest {media_type}"
            ])
        
        return queries
    
    def generate_queries_for_asset(self, row: pd.Series, max_queries: int = 12) -> List[str]:
        """Generate all types of queries for a single asset"""
        all_queries = []
        
        # Extract basic info
        title = self.clean_text(row['title'])
        content_type = row['type']
        actors = self.extract_main_actors(row.get('cast', ''))
        director = self.clean_text(row.get('director', ''))
        genres = self.extract_genres(row.get('listed_in', ''))
        release_year = row.get('release_year') if pd.notna(row.get('release_year')) else None
        
        # Generate different types of queries
        all_queries.extend(self.generate_title_queries(title))
        all_queries.extend(self.generate_actor_queries(actors, content_type))
        all_queries.extend(self.generate_director_queries(director, content_type))
        all_queries.extend(self.generate_genre_queries(genres, release_year))
        
        if release_year:
            all_queries.extend(self.generate_year_queries(int(release_year), content_type))
        
        # Clean and deduplicate
        cleaned_queries = []
        seen = set()
        
        for query in all_queries:
            if query and len(query) > 2:
                # Clean query
                query = re.sub(r'\s+', ' ', query).strip().lower()
                
                # Avoid too generic queries
                if len(query) > 3 and query not in seen:
                    generic_terms = {'movies', 'shows', 'films', 'netflix', 'new', 'best'}
                    if query not in generic_terms:
                        cleaned_queries.append(query)
                        seen.add(query)
        
        # Limit number of queries
        return cleaned_queries[:max_queries]
    
    def generate_ground_truth(self, csv_path: str, output_path: str = None, 
                            sample_size: int = None) -> Dict[str, List[str]]:
        """Generate comprehensive ground truth dataset"""
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path, encoding='latin-1')
        df = df.fillna('')
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            print(f"Sampled {len(df)} assets")
        else:
            print(f"Processing {len(df)} assets")
        
        ground_truth = {}
        processed = 0
        
        for _, row in df.iterrows():
            if processed % 1000 == 0 and processed > 0:
                print(f"Processed {processed}/{len(df)} assets...")
            
            show_id = row.get('show_id') or str(row.iloc[0])  # Fallback to first column
            queries = self.generate_queries_for_asset(row)
            
            if queries:
                ground_truth[show_id] = queries
            
            processed += 1
        
        print(f"Generated ground truth for {len(ground_truth)} assets")
        print(f"Total queries: {sum(len(queries) for queries in ground_truth.values())}")
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(ground_truth, f, indent=2)
            print(f"Saved to: {output_path}")
        
        return ground_truth
    
    def create_evaluation_subset(self, ground_truth: Dict[str, List[str]], 
                               subset_size: int = 1000) -> Dict[str, List[str]]:
        """Create a manageable subset for evaluation"""
        asset_ids = list(ground_truth.keys())
        selected_assets = random.sample(asset_ids, min(subset_size, len(asset_ids)))
        
        subset = {asset_id: ground_truth[asset_id] for asset_id in selected_assets}
        
        print(f"Created evaluation subset:")
        print(f"  Assets: {len(subset):,}")
        print(f"  Total queries: {sum(len(queries) for queries in subset.values()):,}")
        
        return subset