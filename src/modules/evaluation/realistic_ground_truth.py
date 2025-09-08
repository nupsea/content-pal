"""
Realistic Ground Truth Generator for Netflix Content Search

Creates unbiased evaluation data that reflects actual user search patterns
instead of artificially generated queries that favor exact field matching.
"""

import json
import random
import pandas as pd
from typing import Dict, List, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SearchPattern:
    """Represents a realistic search pattern"""
    template: str
    intent: str
    expected_matches: List[str]  # Fields that should match
    weight: float  # How common this pattern is


class RealisticGroundTruthGenerator:
    """
    Generates realistic ground truth based on actual user search patterns
    """
    
    def __init__(self):
        # Real user search patterns based on streaming service usage studies
        self.search_patterns = {
            # Genre + Temporal (very common)
            "genre_temporal": [
                SearchPattern("action movies from {decade}", "genre_time", ["listed_in", "release_year"], 0.15),
                SearchPattern("recent {genre} shows", "genre_time", ["listed_in", "release_year"], 0.12),
                SearchPattern("classic {genre} films", "genre_time", ["listed_in", "release_year"], 0.08),
                SearchPattern("new {genre} series", "genre_time", ["listed_in", "release_year"], 0.10),
                SearchPattern("{genre} movies after {year}", "genre_time", ["listed_in", "release_year"], 0.09),
            ],
            
            # Mood-based searches (increasingly popular)
            "mood_based": [
                SearchPattern("feel-good {genre}", "mood", ["listed_in", "description"], 0.08),
                SearchPattern("dark {genre} shows", "mood", ["listed_in", "description"], 0.06),
                SearchPattern("family-friendly {genre}", "mood", ["listed_in", "description"], 0.07),
                SearchPattern("binge-worthy series", "mood", ["description", "type"], 0.05),
                SearchPattern("award-winning {genre}", "mood", ["listed_in", "description"], 0.04),
                SearchPattern("funny {genre} for date night", "mood", ["listed_in", "description"], 0.03),
            ],
            
            # Actor-based (but not exact names)
            "actor_based": [
                SearchPattern("movies with {popular_actor}", "actor", ["cast"], 0.06),
                SearchPattern("shows starring {tv_actor}", "actor", ["cast"], 0.05),
                SearchPattern("{actor} comedy movies", "actor_genre", ["cast", "listed_in"], 0.04),
                SearchPattern("films featuring {ensemble_cast}", "actor", ["cast"], 0.03),
            ],
            
            # Content theme searches (descriptive)
            "thematic": [
                SearchPattern("movies about {theme}", "theme", ["description", "title"], 0.08),
                SearchPattern("shows based on {theme}", "theme", ["description"], 0.06),
                SearchPattern("stories about {character_type}", "theme", ["description"], 0.05),
                SearchPattern("{theme} documentaries", "theme", ["listed_in", "description"], 0.04),
                SearchPattern("series featuring {relationship_type}", "theme", ["description"], 0.03),
            ],
            
            # Specific but natural title searches
            "title_search": [
                SearchPattern("{partial_title}", "title", ["title"], 0.07),
                SearchPattern("something like {similar_title}", "title", ["title", "description"], 0.04),
                SearchPattern("movies similar to {reference_title}", "similarity", ["description", "listed_in"], 0.03),
            ],
            
            # Director/creator searches (less common but important)
            "creator": [
                SearchPattern("movies by {director}", "director", ["director"], 0.03),
                SearchPattern("{director} films", "director", ["director"], 0.02),
                SearchPattern("shows created by {creator}", "director", ["director"], 0.02),
            ]
        }
        
        # Real search terms based on Netflix/streaming analytics
        self.genre_terms = [
            "comedy", "drama", "thriller", "action", "horror", "romance", "documentary",
            "sci-fi", "fantasy", "crime", "mystery", "animation", "family", "adventure"
        ]
        
        self.mood_descriptors = [
            "feel-good", "dark", "light-hearted", "intense", "uplifting", "gritty",
            "heartwarming", "suspenseful", "emotional", "funny", "serious", "quirky"
        ]
        
        self.themes = [
            "friendship", "love", "betrayal", "revenge", "redemption", "coming of age",
            "family", "war", "crime", "politics", "technology", "survival", "mystery",
            "mental health", "social issues", "historical events", "true crime"
        ]
        
        self.decades = ["90s", "2000s", "2010s", "recent", "last decade"]
        self.years = ["2018", "2019", "2020", "2021", "2022", "2023"]
        
        # Character types people search for
        self.character_types = [
            "strong women", "antiheroes", "detectives", "doctors", "lawyers", "teachers",
            "teenagers", "families", "couples", "friends", "siblings", "outsiders"
        ]
        
        self.relationship_types = [
            "romance", "friendship", "family dynamics", "workplace relationships",
            "forbidden love", "rivalry", "mentorship", "betrayal"
        ]
    
    def load_content_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess Netflix content data"""
        df = pd.read_csv(csv_path, encoding='latin-1').fillna('')
        
        # Extract popular actors (those appearing in multiple shows)
        all_cast = []
        for cast_list in df['cast'].dropna():
            if cast_list.strip():
                actors = [actor.strip() for actor in cast_list.split(',')[:3]]  # Top 3 actors
                all_cast.extend(actors)
        
        # Get actors who appear in at least 2 shows (more likely to be searched for)
        actor_counts = pd.Series(all_cast).value_counts()
        self.popular_actors = actor_counts[actor_counts >= 2].head(50).index.tolist()
        
        # Extract popular directors
        director_counts = df['director'].value_counts()
        self.popular_directors = director_counts[director_counts >= 2].head(30).index.tolist()
        
        # Get diverse title samples for partial matching
        self.sample_titles = df['title'].sample(min(100, len(df))).tolist()
        
        return df
    
    def generate_realistic_query(self, content_row: pd.Series, pattern: SearchPattern) -> Tuple[str, List[str]]:
        """Generate a realistic query based on content and pattern"""
        
        # Fill template with realistic values
        query = pattern.template
        relevant_ids = []
        
        # Replace placeholders based on pattern type
        if "{genre}" in query:
            # Extract genre from listed_in field
            genres = content_row['listed_in'].lower()
            matching_genre = None
            for genre in self.genre_terms:
                if genre in genres or any(g in genres for g in [genre + 's', genre + 'ies']):
                    matching_genre = genre
                    break
            
            if matching_genre:
                query = query.replace("{genre}", matching_genre)
                # Find other content with this genre
                relevant_ids.append(content_row['show_id'])
            else:
                return None, []
        
        if "{decade}" in query or "{year}" in query:
            year = content_row['release_year']
            if pd.isna(year) or year == '':
                return None, []
                
            year = int(year) if isinstance(year, (int, float)) else int(year)
            
            if "{decade}" in query:
                if year >= 2020:
                    decade = "recent"
                elif year >= 2010:
                    decade = "2010s" 
                elif year >= 2000:
                    decade = "2000s"
                elif year >= 1990:
                    decade = "90s"
                else:
                    decade = "classic"
                query = query.replace("{decade}", decade)
            
            if "{year}" in query:
                base_year = max(2000, year - 5)  # Search for movies after a reasonable year
                query = query.replace("{year}", str(base_year))
        
        if "{popular_actor}" in query or "{tv_actor}" in query or "{actor}" in query:
            cast = content_row['cast']
            if pd.isna(cast) or not cast.strip():
                return None, []
                
            actors = [actor.strip() for actor in cast.split(',')[:2]]
            matching_actor = None
            
            for actor in actors:
                if actor in self.popular_actors:
                    matching_actor = actor
                    break
            
            if matching_actor:
                query = query.replace("{popular_actor}", matching_actor)
                query = query.replace("{tv_actor}", matching_actor) 
                query = query.replace("{actor}", matching_actor)
                relevant_ids.append(content_row['show_id'])
            else:
                return None, []
        
        if "{director}" in query or "{creator}" in query:
            director = content_row['director']
            if pd.isna(director) or not director.strip():
                return None, []
                
            if director in self.popular_directors:
                query = query.replace("{director}", director)
                query = query.replace("{creator}", director)
                relevant_ids.append(content_row['show_id'])
            else:
                return None, []
        
        if "{theme}" in query:
            theme = random.choice(self.themes)
            query = query.replace("{theme}", theme)
            # For thematic searches, the current item should be relevant if description matches
            if theme.lower() in content_row['description'].lower():
                relevant_ids.append(content_row['show_id'])
        
        if "{character_type}" in query:
            char_type = random.choice(self.character_types)
            query = query.replace("{character_type}", char_type)
        
        if "{relationship_type}" in query:
            rel_type = random.choice(self.relationship_types)
            query = query.replace("{relationship_type}", rel_type)
        
        if "{partial_title}" in query:
            title = content_row['title']
            if len(title.split()) > 1:
                # Take first 1-2 words of title for partial search
                partial = ' '.join(title.split()[:2])
                query = query.replace("{partial_title}", partial)
                relevant_ids.append(content_row['show_id'])
            else:
                return None, []
        
        if "{similar_title}" in query or "{reference_title}" in query:
            ref_title = random.choice(self.sample_titles)
            query = query.replace("{similar_title}", ref_title)
            query = query.replace("{reference_title}", ref_title)
        
        # Clean up any remaining placeholders
        if "{" in query and "}" in query:
            return None, []
        
        return query, relevant_ids
    
    def find_relevant_content(self, df: pd.DataFrame, query: str, pattern: SearchPattern, 
                            primary_id: str) -> List[str]:
        """Find content that should match the query"""
        relevant_ids = [primary_id]  # Always include the primary item
        
        # Extract search criteria from query
        query_lower = query.lower()
        
        # Genre matching
        if "listed_in" in pattern.expected_matches:
            primary_genres = df[df['show_id'] == primary_id]['listed_in'].iloc[0].lower()
            for _, row in df.iterrows():
                if row['show_id'] != primary_id:
                    content_genres = row['listed_in'].lower()
                    # Check for genre overlap
                    if any(genre.strip() in content_genres for genre in primary_genres.split(',') if genre.strip()):
                        # Additional filters based on query
                        if self._matches_additional_criteria(query_lower, row):
                            relevant_ids.append(row['show_id'])
                            if len(relevant_ids) >= 20:  # Limit to prevent huge lists
                                break
        
        # Actor matching
        if "cast" in pattern.expected_matches:
            primary_cast = df[df['show_id'] == primary_id]['cast'].iloc[0]
            if pd.notna(primary_cast):
                primary_actors = [a.strip() for a in primary_cast.split(',')[:2]]
                for _, row in df.iterrows():
                    if row['show_id'] != primary_id and pd.notna(row['cast']):
                        content_cast = row['cast']
                        if any(actor in content_cast for actor in primary_actors):
                            relevant_ids.append(row['show_id'])
                            if len(relevant_ids) >= 15:
                                break
        
        # Director matching  
        if "director" in pattern.expected_matches:
            primary_director = df[df['show_id'] == primary_id]['director'].iloc[0]
            if pd.notna(primary_director):
                for _, row in df.iterrows():
                    if row['show_id'] != primary_id and row['director'] == primary_director:
                        relevant_ids.append(row['show_id'])
                        if len(relevant_ids) >= 10:
                            break
        
        return relevant_ids[:25]  # Cap at 25 relevant items per query
    
    def _matches_additional_criteria(self, query_lower: str, content_row: pd.Series) -> bool:
        """Check if content matches additional query criteria"""
        
        # Year filtering
        if "after" in query_lower:
            year_match = None
            for word in query_lower.split():
                if word.isdigit() and len(word) == 4:
                    year_match = int(word)
                    break
            if year_match and content_row['release_year']:
                try:
                    content_year = int(content_row['release_year'])
                    if content_year <= year_match:
                        return False
                except (ValueError, TypeError):
                    pass
        
        # Decade filtering
        if "recent" in query_lower and content_row['release_year']:
            try:
                content_year = int(content_row['release_year'])
                if content_year < 2020:
                    return False
            except (ValueError, TypeError):
                pass
                
        if "classic" in query_lower and content_row['release_year']:
            try:
                content_year = int(content_row['release_year'])
                if content_year > 2000:
                    return False
            except (ValueError, TypeError):
                pass
        
        # Content type filtering
        if "movie" in query_lower and content_row['type'].lower() != 'movie':
            return False
        if ("show" in query_lower or "series" in query_lower) and content_row['type'].lower() != 'tv show':
            return False
        
        # Mood filtering (basic)
        description = content_row['description'].lower()
        if "dark" in query_lower:
            if not any(word in description for word in ['dark', 'grit', 'brutal', 'violent', 'crime']):
                return False
        
        if "family" in query_lower or "kid" in query_lower:
            if any(word in description for word in ['violent', 'brutal', 'murder', 'killing']):
                return False
                
        return True
    
    def generate_realistic_ground_truth(self, csv_path: str, output_path: str, 
                                      num_queries: int = 1000) -> Dict[str, List[str]]:
        """Generate comprehensive realistic ground truth"""
        
        print(f"📊 Generating realistic ground truth with {num_queries} queries...")
        
        # Load content data
        df = self.load_content_data(csv_path)
        print(f"✅ Loaded {len(df)} content items")
        print(f"✅ Found {len(self.popular_actors)} popular actors")
        print(f"✅ Found {len(self.popular_directors)} popular directors")
        
        ground_truth = {}
        patterns_used = defaultdict(int)
        
        # Calculate pattern distribution
        total_weight = sum(sum(p.weight for p in patterns) for patterns in self.search_patterns.values())
        
        queries_generated = 0
        attempts = 0
        max_attempts = num_queries * 3  # Prevent infinite loop
        
        while queries_generated < num_queries and attempts < max_attempts:
            attempts += 1
            
            # Select pattern category based on weights
            category = self._select_pattern_category()
            pattern = random.choices(
                self.search_patterns[category], 
                weights=[p.weight for p in self.search_patterns[category]]
            )[0]
            
            # Select random content item
            content_row = df.sample(1).iloc[0]
            
            # Generate query
            query, primary_ids = self.generate_realistic_query(content_row, pattern)
            
            if query and primary_ids:
                # Find all relevant content
                relevant_ids = self.find_relevant_content(df, query, pattern, primary_ids[0])
                
                if len(relevant_ids) >= 2:  # Need at least 2 relevant items
                    ground_truth[query] = relevant_ids
                    patterns_used[f"{category}:{pattern.template}"] += 1
                    queries_generated += 1
                    
                    if queries_generated % 100 == 0:
                        print(f"Generated {queries_generated}/{num_queries} queries...")
        
        print(f"✅ Generated {len(ground_truth)} realistic queries")
        
        # Print pattern distribution
        print(f"\n📈 Pattern Distribution:")
        for pattern, count in sorted(patterns_used.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {pattern}: {count} queries")
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"💾 Saved realistic ground truth to {output_path}")
        return ground_truth
    
    def _select_pattern_category(self) -> str:
        """Select pattern category based on realistic usage weights"""
        categories = list(self.search_patterns.keys())
        category_weights = [
            sum(p.weight for p in self.search_patterns[cat]) 
            for cat in categories
        ]
        return random.choices(categories, weights=category_weights)[0]


def main():
    """Generate realistic ground truth for evaluation"""
    
    generator = RealisticGroundTruthGenerator()
    
    # Generate realistic ground truth
    ground_truth = generator.generate_realistic_ground_truth(
        csv_path="../data/netflix_titles_cleaned.csv",
        output_path="../realistic_ground_truth.json",
        num_queries=800  # More manageable size
    )
    
    # Print sample queries
    print(f"\n🔍 Sample Realistic Queries:")
    for i, (query, relevant_ids) in enumerate(list(ground_truth.items())[:10]):
        print(f"{i+1:2d}. '{query}' -> {len(relevant_ids)} relevant items")


if __name__ == "__main__":
    main()