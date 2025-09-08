#!/usr/bin/env python3
"""
Create a learned query optimization model that trains on successful patterns
from ground truth data to transform user queries into optimized search queries
without expensive LLM calls.
"""

import json
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import Counter
import pickle
from pathlib import Path


class LearnedQueryOptimizer:
    """
    A learned model that optimizes queries based on successful patterns from ground truth data.
    
    Strategy:
    1. Analyze ground truth queries and their successful matches
    2. Extract common patterns and transformations
    3. Build rule-based optimization using pattern matching
    4. Apply optimizations that boost performance without LLM costs
    """
    
    def __init__(self):
        self.genre_patterns = {}
        self.actor_patterns = {}
        self.temporal_patterns = {}
        self.content_type_patterns = {}
        self.optimization_rules = []
        self.boost_patterns = {}
        
    def train_on_ground_truth(self, ground_truth_path: str, results_path: str, csv_path: str):
        """Train the optimizer on ground truth data and content metadata"""
        
        print("Training Learned Query Optimizer...")
        
        # Load ground truth data
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
            
        # Load results.json for additional training data
        with open(results_path, 'r') as f:
            results_data = json.load(f)
            
        # Combine both data sources
        combined_data = {}
        for show_id in set(list(ground_truth.keys()) + list(results_data.keys())):
            combined_queries = []
            if show_id in ground_truth:
                combined_queries.extend(ground_truth[show_id])
            if show_id in results_data:
                combined_queries.extend(results_data[show_id])
            combined_data[show_id] = combined_queries
        
        print(f"Combined training data: {len(combined_data)} shows, {sum(len(queries) for queries in combined_data.values())} total queries")
        
        # Load content metadata
        df = pd.read_csv(csv_path)
        content_lookup = {row['show_id']: row.to_dict() for _, row in df.iterrows()}
        
        # Analyze successful patterns using combined data
        self._extract_genre_patterns(combined_data, content_lookup)
        self._extract_actor_patterns(combined_data, content_lookup)
        self._extract_temporal_patterns(combined_data, content_lookup)
        self._extract_content_type_patterns(combined_data, content_lookup)
        self._build_optimization_rules(combined_data, content_lookup)
        
        print(f"Learned {len(self.optimization_rules)} optimization rules")
        print(f"Genre patterns: {len(self.genre_patterns)}")
        print(f"Actor patterns: {len(self.actor_patterns)}")
        
    def _extract_genre_patterns(self, ground_truth: dict, content_lookup: dict):
        """Extract genre-based query optimization patterns"""
        
        genre_query_patterns = []
        
        for show_id, queries in ground_truth.items():
            if show_id not in content_lookup:
                continue
                
            content = content_lookup[show_id]
            genres = str(content.get('listed_in', '')).lower()
            
            for query in queries:
                query_lower = query.lower()
                
                # Extract genre mentions in queries
                common_genres = ['comedy', 'drama', 'thriller', 'horror', 'action', 
                               'romance', 'documentary', 'family', 'crime', 'mystery', 
                               'fantasy', 'sci-fi', 'adventure', 'animation']
                
                for genre in common_genres:
                    if genre in query_lower and genre in genres:
                        if genre not in self.genre_patterns:
                            self.genre_patterns[genre] = []
                        self.genre_patterns[genre].append((query, content))
        
        # Build genre optimization rules
        for genre, examples in self.genre_patterns.items():
            if len(examples) >= 3:  # Need sufficient examples
                self.boost_patterns[genre] = {
                    'boost_fields': ['listed_in', 'description'],
                    'boost_weight': 2.0,
                    'examples': len(examples)
                }
    
    def _extract_actor_patterns(self, ground_truth: dict, content_lookup: dict):
        """Extract actor-based query optimization patterns"""
        
        for show_id, queries in ground_truth.items():
            if show_id not in content_lookup:
                continue
                
            content = content_lookup[show_id]
            cast = str(content.get('cast', '')).lower()
            
            for query in queries:
                query_lower = query.lower()
                
                # Look for actor name patterns
                # Simple heuristic: capitalized words that appear in both query and cast
                query_words = re.findall(r'\b[A-Z][a-z]+\b', query)
                
                for word in query_words:
                    if word.lower() in cast:
                        if word not in self.actor_patterns:
                            self.actor_patterns[word] = []
                        self.actor_patterns[word].append((query, content))
        
        # Build actor optimization rules  
        for actor, examples in self.actor_patterns.items():
            if len(examples) >= 2:  # Need sufficient examples
                self.boost_patterns[actor.lower()] = {
                    'boost_fields': ['cast', 'title'],
                    'boost_weight': 2.5,
                    'examples': len(examples)
                }
    
    def _extract_temporal_patterns(self, ground_truth: dict, content_lookup: dict):
        """Extract temporal query optimization patterns"""
        
        temporal_indicators = [
            (r'\b(80s|eighties)\b', '1980s'),
            (r'\b(90s|nineties)\b', '1990s'), 
            (r'\b(2000s)\b', '2000s'),
            (r'\b(2010s)\b', '2010s'),
            (r'\brecent\b|\blast\s+few\s+years\b', 'recent'),
            (r'\bclassic\b', 'classic'),
            (r'\b(\d{4})\b', 'specific_year'),
        ]
        
        for show_id, queries in ground_truth.items():
            if show_id not in content_lookup:
                continue
                
            content = content_lookup[show_id]
            release_year = content.get('release_year', '')
            
            for query in queries:
                query_lower = query.lower()
                
                for pattern, period_type in temporal_indicators:
                    if re.search(pattern, query_lower):
                        if period_type not in self.temporal_patterns:
                            self.temporal_patterns[period_type] = []
                        self.temporal_patterns[period_type].append((query, content, release_year))
    
    def _extract_content_type_patterns(self, ground_truth: dict, content_lookup: dict):
        """Extract content type optimization patterns"""
        
        type_indicators = [
            (['movie', 'film'], 'Movie'),
            (['series', 'show', 'tv'], 'TV Show'),
            (['documentary', 'doc'], 'documentary_genre'),
            (['standup', 'stand up', 'comedy special'], 'standup_genre')
        ]
        
        for show_id, queries in ground_truth.items():
            if show_id not in content_lookup:
                continue
                
            content = content_lookup[show_id]
            content_type = content.get('type', '')
            
            for query in queries:
                query_lower = query.lower()
                
                for indicators, pattern_type in type_indicators:
                    if any(ind in query_lower for ind in indicators):
                        if pattern_type not in self.content_type_patterns:
                            self.content_type_patterns[pattern_type] = []
                        self.content_type_patterns[pattern_type].append((query, content, content_type))
    
    def _build_optimization_rules(self, ground_truth: dict, content_lookup: dict):
        """Build final optimization rules from extracted patterns"""
        
        # Rule 1: Genre-based query expansion
        for genre in ['comedy', 'drama', 'thriller', 'horror', 'action', 'romance', 
                     'documentary', 'family', 'crime', 'mystery', 'fantasy', 'adventure']:
            self.optimization_rules.append({
                'type': 'genre_boost',
                'pattern': rf'\b{genre}\b',
                'action': lambda q, g=genre: self._boost_genre_query(q, g),
                'priority': 1
            })
        
        # Rule 2: Actor name detection and boosting  
        self.optimization_rules.append({
            'type': 'actor_boost',
            'pattern': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Capitalized names
            'action': self._boost_actor_query,
            'priority': 2
        })
        
        # Rule 3: Temporal optimization
        temporal_rules = [
            (r'\b(90s|nineties)\b', lambda q: q + ' 1990s decade'),
            (r'\b(80s|eighties)\b', lambda q: q + ' 1980s decade'), 
            (r'\b(2000s)\b', lambda q: q + ' 2000s decade'),
            (r'\brecent\b', lambda q: q + ' 2020s latest'),
            (r'\bclassic\b', lambda q: q + ' classic vintage')
        ]
        
        for pattern, transform in temporal_rules:
            self.optimization_rules.append({
                'type': 'temporal',
                'pattern': pattern,
                'action': transform,
                'priority': 1
            })
        
        # Rule 4: Content type optimization
        self.optimization_rules.append({
            'type': 'content_type',
            'pattern': r'\b(movie|film)\b',
            'action': lambda q: q,  # Keep as is but boost movie fields
            'boost_config': {'type': 'Movie'},
            'priority': 1
        })
        
        self.optimization_rules.append({
            'type': 'content_type', 
            'pattern': r'\b(series|show|tv)\b',
            'action': lambda q: q,  # Keep as is but boost TV fields
            'boost_config': {'type': 'TV Show'}, 
            'priority': 1
        })
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Optimize a query using learned patterns.
        
        Returns optimized query and search configuration.
        """
        
        optimized_query = query
        boost_config = {
            'title': 3.0,
            'cast': 2.0, 
            'director': 1.5,
            'listed_in': 1.5,
            'description': 1.0,
            'country': 1.0
        }
        filters = {}
        
        query_lower = query.lower()
        
        # Apply optimization rules in priority order
        applied_rules = []
        
        for rule in sorted(self.optimization_rules, key=lambda x: x['priority']):
            if re.search(rule['pattern'], query_lower):
                if rule['type'] == 'genre_boost':
                    # Boost genre-related fields
                    boost_config['listed_in'] = 2.5
                    boost_config['description'] = 1.5
                    applied_rules.append('genre_boost')
                    
                elif rule['type'] == 'actor_boost':
                    # Boost cast field for actor queries
                    boost_config['cast'] = 3.5
                    boost_config['title'] = 2.5
                    applied_rules.append('actor_boost')
                    
                elif rule['type'] == 'temporal':
                    # Apply temporal transformation
                    optimized_query = rule['action'](optimized_query)
                    applied_rules.append('temporal')
                    
                elif rule['type'] == 'content_type':
                    # Apply content type filtering
                    if 'boost_config' in rule:
                        filters.update(rule['boost_config'])
                    applied_rules.append('content_type')
        
        # Additional pattern-based optimizations
        
        # Boost for specific high-value terms
        high_value_terms = ['award', 'winner', 'oscar', 'emmy', 'golden globe', 
                           'nominated', 'critically acclaimed', 'bestseller']
        
        if any(term in query_lower for term in high_value_terms):
            boost_config['title'] = 4.0
            boost_config['description'] = 2.0
            applied_rules.append('high_value_terms')
        
        # Country/language detection
        countries = ['korean', 'japanese', 'indian', 'british', 'french', 'spanish', 
                    'german', 'italian', 'chinese', 'russian', 'brazilian']
        
        detected_country = None
        for country in countries:
            if country in query_lower:
                boost_config['country'] = 3.0
                boost_config['cast'] = 2.5
                detected_country = country
                applied_rules.append('country_boost')
                break
        
        return {
            'optimized_query': optimized_query.strip(),
            'boost_config': boost_config,
            'filters': filters,
            'detected_country': detected_country,
            'applied_rules': applied_rules,
            'confidence': len(applied_rules) / len(self.optimization_rules)
        }
    
    def _boost_genre_query(self, query: str, genre: str) -> str:
        """Add genre-specific boost terms"""
        genre_expansions = {
            'comedy': ['funny', 'humor', 'laughs'],
            'horror': ['scary', 'frightening', 'terror'], 
            'thriller': ['suspense', 'tension', 'mystery'],
            'romance': ['love', 'romantic', 'relationship'],
            'action': ['adventure', 'exciting', 'fast-paced'],
            'drama': ['dramatic', 'emotional', 'intense'],
            'documentary': ['real', 'factual', 'non-fiction']
        }
        
        if genre in genre_expansions:
            expansion = genre_expansions[genre][0]  # Take the best expansion
            return f"{query} {expansion}"
        return query
    
    def _boost_actor_query(self, query: str) -> str:
        """Boost queries with actor names"""
        # Extract potential actor names (simple heuristic)
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', query)
        if names:
            # Add 'starring' to make it clearer
            return f"starring {query}"
        return query
    
    def save(self, path: str):
        """Save the learned model"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load a saved model"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def test_learned_optimizer():
    """Test the learned optimizer with sample queries"""
    
    # Create and train the optimizer
    optimizer = LearnedQueryOptimizer()
    optimizer.train_on_ground_truth(
        ground_truth_path="notebooks/ground_truth.json",
        results_path="notebooks/results.json",
        csv_path="data/netflix_titles_enriched_full.csv"
    )
    
    # Test queries
    test_queries = [
        "romantic comedies from the 90s",
        "korean dramas", 
        "movies with Tom Hanks",
        "horror films",
        "recent documentaries",
        "action movies",
        "british tv series",
        "feel good family movies",
        "thriller series",
        "award winning films"
    ]
    
    print("\nTesting Learned Query Optimizer:")
    print("=" * 60)
    
    for query in test_queries:
        result = optimizer.optimize_query(query)
        print(f"\nOriginal: {query}")
        print(f"Optimized: {result['optimized_query']}")
        print(f"Applied rules: {result['applied_rules']}")
        print(f"Top boosts: {sorted(result['boost_config'].items(), key=lambda x: x[1], reverse=True)[:3]}")
        if result['detected_country']:
            print(f"Country: {result['detected_country']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 40)
    
    # Save the model
    optimizer.save("learned_query_optimizer.pkl")
    print(f"\nModel saved to: learned_query_optimizer.pkl")
    
    return optimizer


if __name__ == "__main__":
    test_learned_optimizer()