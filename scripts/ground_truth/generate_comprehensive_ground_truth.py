#!/usr/bin/env python3
"""
Generate comprehensive ground truth dataset with realistic queries for every Netflix asset.
Creates multiple query types that users would actually search for.
"""

import json
import pandas as pd
import re
import random
from typing import List, Dict, Set, Optional
from collections import defaultdict

class GroundTruthGenerator:
    def __init__(self):
        # Popular actor names that appear frequently in searches
        self.popular_actors = set()
        
        # Genre mappings for more natural queries
        self.genre_mappings = {
            'Action & Adventure': ['action movies', 'adventure films', 'action adventure'],
            'Comedies': ['comedy movies', 'funny movies', 'comedies', 'comedy films'],
            'Documentaries': ['documentaries', 'documentary films', 'docs'],
            'Dramas': ['drama movies', 'drama films', 'dramas'],
            'Horror': ['horror movies', 'scary movies', 'horror films'],
            'Romantic Movies': ['romantic movies', 'romance films', 'love movies'],
            'Thrillers': ['thriller movies', 'thriller films', 'thrillers'],
            'Sci-Fi & Fantasy': ['sci-fi movies', 'fantasy films', 'science fiction'],
            'Crime': ['crime movies', 'crime shows', 'criminal dramas'],
            'TV Dramas': ['tv dramas', 'drama series', 'drama shows'],
            'TV Comedies': ['tv comedies', 'comedy series', 'sitcoms'],
            'Reality TV': ['reality shows', 'reality tv', 'reality series'],
            'International Movies': ['foreign movies', 'international films'],
            'International TV Shows': ['foreign shows', 'international series'],
            'Children & Family Movies': ['kids movies', 'family films', 'children movies'],
            'British TV Shows': ['british shows', 'uk series', 'british tv'],
            'Korean TV Shows': ['k-dramas', 'korean series', 'korean shows'],
            'Spanish-Language TV Shows': ['spanish shows', 'latino series'],
            'Anime': ['anime series', 'anime shows', 'anime movies']
        }
        
        # Common search patterns
        self.year_ranges = {
            '2020s': [2020, 2021, 2022, 2023, 2024],
            '2010s': [2015, 2016, 2017, 2018, 2019],
            'recent': [2020, 2021, 2022, 2023],
            'new': [2021, 2022, 2023]
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or text == '':
            return ''
        return str(text).strip()

    def extract_main_actors(self, cast_str: str, max_actors: int = 3) -> List[str]:
        """Extract main actors (first few in cast list)"""
        cast = self.clean_text(cast_str)
        if not cast:
            return []
        
        actors = []
        for actor in cast.split(',')[:max_actors]:
            actor = actor.strip()
            # Filter out actors with special characters or very short names
            if len(actor) > 5 and ' ' in actor and actor.replace(' ', '').replace("'", "").replace("-", "").isalpha():
                actors.append(actor)
        return actors

    def extract_genres(self, listed_in: str) -> List[str]:
        """Extract and normalize genres"""
        genres_str = self.clean_text(listed_in)
        if not genres_str:
            return []
        
        genres = []
        for genre in genres_str.split(','):
            genre = genre.strip()
            if genre and len(genre) > 2:
                genres.append(genre)
        return genres

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
                f"{actor} netflix",
                f"what {media_type} has {actor}"
            ])
        
        return queries

    def generate_title_queries(self, title: str, content_type: str) -> List[str]:
        """Generate title-based queries"""
        if not title:
            return []
            
        queries = [title]  # Exact title
        
        # Add variations
        words = title.split()
        if len(words) >= 2:
            # First few words
            queries.append(' '.join(words[:2]))
            if len(words) >= 3:
                queries.append(' '.join(words[:3]))
        
        # Add type qualifiers
        media_type = 'movie' if content_type == 'Movie' else 'show'
        queries.extend([
            f"{title} {media_type}",
            f"{title} netflix",
            f"watch {title}"
        ])
        
        # Add partial matches for longer titles
        if len(words) > 3:
            queries.append(words[0])  # First word only
        
        return queries

    def generate_genre_queries(self, genres: List[str], release_year: Optional[int], content_type: str) -> List[str]:
        """Generate genre-based queries"""
        queries = []
        
        for genre in genres[:2]:  # Use top 2 genres
            # Map to more natural query terms
            natural_queries = self.genre_mappings.get(genre, [genre.lower()])
            
            for natural_genre in natural_queries:
                if release_year is not None:
                    queries.extend([
                        natural_genre,
                        f"{natural_genre} {release_year}",
                        f"best {natural_genre}",
                        f"new {natural_genre}",
                        f"{natural_genre} netflix"
                    ])
                    
                    # Add decade-based queries
                    if 2020 <= release_year <= 2024:
                        queries.append(f"{natural_genre} 2020s")
                    elif 2010 <= release_year <= 2019:
                        queries.append(f"{natural_genre} 2010s")
                else:
                    # Skip year-specific queries when release_year is missing
                    queries.extend([
                        natural_genre,
                        f"best {natural_genre}",
                        f"new {natural_genre}",
                        f"{natural_genre} netflix"
                    ])
        
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

    def generate_description_queries(self, description: str, genres: List[str]) -> List[str]:
        """Generate content/plot-based queries"""
        desc = self.clean_text(description).lower()
        if not desc:
            return []
        
        queries = []
        
        # Extract key themes and concepts
        theme_patterns = {
            'family': ['family', 'father', 'mother', 'son', 'daughter', 'parents'],
            'love': ['love', 'romance', 'relationship', 'dating', 'marriage'],
            'crime': ['crime', 'murder', 'detective', 'police', 'criminal'],
            'war': ['war', 'battle', 'military', 'soldier', 'combat'],
            'school': ['school', 'student', 'college', 'university', 'teacher'],
            'supernatural': ['magic', 'vampire', 'zombie', 'ghost', 'supernatural'],
            'travel': ['journey', 'travel', 'adventure', 'explore'],
            'business': ['business', 'company', 'corporate', 'entrepreneur'],
            'sports': ['football', 'basketball', 'soccer', 'baseball', 'sport'],
            'music': ['music', 'band', 'singer', 'musician', 'concert']
        }
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in desc for keyword in keywords):
                queries.extend([
                    f"{theme} movies",
                    f"films about {theme}",
                    f"{theme} stories"
                ])
        
        return queries

    def generate_year_queries(self, release_year: int, content_type: str) -> List[str]:
        """Generate year-based queries"""
        if pd.isna(release_year):
            return []
        
        year = int(release_year)
        media_type = 'movies' if content_type == 'Movie' else 'shows'
        
        queries = [
            f"{year} {media_type}",
            f"best {media_type} {year}",
            f"{year} films"
        ]
        
        # Add recent/new qualifiers for recent content
        if year >= 2020:
            queries.extend([
                f"new {media_type}",
                f"recent {media_type}",
                f"latest {media_type}"
            ])
        
        return queries

    def generate_country_queries(self, country: str, content_type: str) -> List[str]:
        """Generate country/international queries"""
        country = self.clean_text(country)
        if not country:
            return []
        
        # Focus on major countries that users search for
        major_countries = {
            'United States': ['american', 'us', 'hollywood'],
            'United Kingdom': ['british', 'uk', 'english'],
            'South Korea': ['korean', 'k-drama', 'korean drama'],
            'Japan': ['japanese', 'anime'],
            'India': ['indian', 'bollywood'],
            'France': ['french'],
            'Germany': ['german'],
            'Spain': ['spanish'],
            'Italy': ['italian'],
            'Canada': ['canadian'],
            'Australia': ['australian']
        }
        
        queries = []
        media_type = 'movies' if content_type == 'Movie' else 'shows'
        
        for main_country in country.split(',')[:1]:  # Use first country
            main_country = main_country.strip()
            
            # Direct country name
            queries.append(f"{main_country.lower()} {media_type}")
            
            # Use mapped terms if available
            if main_country in major_countries:
                for term in major_countries[main_country]:
                    queries.extend([
                        f"{term} {media_type}",
                        f"{term} films"
                    ])
        
        return queries

    def generate_queries_for_asset(self, row: pd.Series) -> List[str]:
        """Generate all types of queries for a single asset"""
        all_queries = []
        
        # Extract basic info
        title = self.clean_text(row['title'])
        content_type = row['type']
        actors = self.extract_main_actors(row['cast'])
        director = self.clean_text(row['director'])
        genres = self.extract_genres(row['listed_in'])
        description = self.clean_text(row['description'])
        release_year = row['release_year'] if pd.notna(row['release_year']) else None
        country = self.clean_text(row['country'])
        
        # Generate different types of queries
        all_queries.extend(self.generate_title_queries(title, content_type))
        all_queries.extend(self.generate_actor_queries(actors, content_type))
        all_queries.extend(self.generate_director_queries(director, content_type))
        all_queries.extend(self.generate_genre_queries(genres, release_year, content_type))
        all_queries.extend(self.generate_description_queries(description, genres))
        
        if release_year:
            all_queries.extend(self.generate_year_queries(release_year, content_type))
        all_queries.extend(self.generate_country_queries(country, content_type))
        
        # Clean and deduplicate
        cleaned_queries = []
        seen = set()
        
        for query in all_queries:
            if query and len(query) > 2:
                # Clean query
                query = re.sub(r'\s+', ' ', query).strip().lower()
                
                # Avoid too generic queries
                if len(query) > 3 and query not in seen:
                    # Avoid queries that are too generic
                    generic_terms = {'movies', 'shows', 'films', 'netflix', 'new', 'best'}
                    if query not in generic_terms:
                        cleaned_queries.append(query)
                        seen.add(query)
        
        # Limit to top queries by priority
        return cleaned_queries[:15]  # Max 15 queries per asset

    def generate_comprehensive_ground_truth(self, csv_file: str, output_file: str):
        """Generate ground truth for all assets"""
        print("Loading Netflix dataset...")
        df = pd.read_csv(csv_file, encoding='latin-1')
        print(f"Loaded {len(df)} assets")
        
        ground_truth = {}
        stats = defaultdict(int)
        
        print("Generating queries for all assets...")
        # Use enumerate to get a guaranteed int counter instead of the DataFrame index (Hashable)
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(df)} assets...")
            
            show_id = row['show_id']
            queries = self.generate_queries_for_asset(row)
            
            if queries:
                ground_truth[show_id] = queries
                stats['assets_with_queries'] += 1
                stats['total_queries'] += len(queries)
            else:
                stats['assets_without_queries'] += 1
        
        print(f"\nGeneration complete!")
        print(f"Assets with queries: {stats['assets_with_queries']}")
        print(f"Assets without queries: {stats['assets_without_queries']}")
        print(f"Total queries generated: {stats['total_queries']}")
        print(f"Average queries per asset: {stats['total_queries']/stats['assets_with_queries']:.1f}")
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"Saved comprehensive ground truth to: {output_file}")
        
        # Show examples
        print(f"\nSample queries:")
        sample_ids = list(ground_truth.keys())[:5]
        for show_id in sample_ids:
            content = df[df['show_id'] == show_id].iloc[0]
            print(f"\n{show_id}: {content['title']} ({content['type']})")
            print(f"Sample queries: {ground_truth[show_id][:5]}")
        
        return ground_truth

def main():
    generator = GroundTruthGenerator()
    
    # Generate comprehensive ground truth
    ground_truth = generator.generate_comprehensive_ground_truth(
        'data/netflix_titles_cleaned.csv',
        'comprehensive_ground_truth.json'
    )
    
    print(f"\n🎉 Successfully generated comprehensive ground truth dataset!")
    print(f"Total assets: {len(ground_truth)}")
    print(f"Total queries: {sum(len(queries) for queries in ground_truth.values())}")

if __name__ == "__main__":
    main()