"""
Strategic Search Enhancement System
A comprehensive approach to achieve >50% performance on realistic queries
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .core import ContentSearchSystem, SearchConfig, SearchResult
from .cross_encoder_reranker import get_cross_encoder_reranker, prepare_cross_encoder_reranking
from ..rag import AdaptiveRetriever


@dataclass
class StrategicSearchConfig:
    """Advanced configuration for strategic search"""
    # Multi-field query expansion
    expand_semantic_fields: bool = True
    expand_temporal_queries: bool = True
    expand_cast_variations: bool = True
    
    # Advanced matching
    use_fuzzy_title_matching: bool = True
    use_description_analysis: bool = True
    use_genre_intelligence: bool = True
    
    # Scoring enhancements
    boost_exact_phrase_match: float = 3.0
    boost_partial_phrase_match: float = 1.8
    boost_multi_field_match: float = 2.2
    
    # Result fusion
    max_results_per_strategy: int = 20
    final_result_count: int = 50
    
    # Cross-encoder reranking
    use_cross_encoder: bool = True
    cross_encoder_weight: float = 0.7


class StrategicSearchSystem:
    """Advanced search system targeting >50% performance on realistic queries"""
    
    def __init__(self, backend_type: str = "minsearch"):
        self.backend_type = backend_type
        
        # Multiple search strategies
        self.primary_search = ContentSearchSystem(backend_type=backend_type)
        self.adaptive_search = AdaptiveRetriever(backend_type=backend_type)
        
        # Advanced field mappings
        self.genre_synonyms = {
            'comedy': ['funny', 'humor', 'humorous', 'hilarious', 'amusing', 'feel good', 'feel-good', 'lighthearted'],
            'drama': ['dramatic', 'serious', 'emotional', 'touching', 'heartfelt', 'powerful'],
            'thriller': ['suspense', 'suspenseful', 'tense', 'edge-of-seat', 'psychological', 'mind-bending'],
            'horror': ['scary', 'frightening', 'terrifying', 'spooky', 'dark', 'creepy', 'chilling'],
            'documentary': ['docs', 'factual', 'real story', 'true story', 'behind the scenes', 'making of'],
            'romance': ['romantic', 'love story', 'relationship', 'love', 'heartwarming'],
            'action': ['adventure', 'exciting', 'fast-paced', 'adrenaline'],
            'family': ['family-friendly', 'kids', 'children', 'wholesome', 'all ages'],
            'international': ['foreign', 'world cinema', 'subtitled', 'from'],
            'independent': ['indie', 'art house', 'low budget', 'festival']
        }
        
        # Mood to genre mapping
        self.mood_to_genre = {
            'feel good': ['comedy', 'family', 'romantic'],
            'dark': ['thriller', 'horror', 'drama'],
            'intense': ['thriller', 'drama', 'action'],
            'uplifting': ['comedy', 'family', 'documentary'],
            'thought-provoking': ['drama', 'documentary', 'international']
        }
        
        # Rating synonyms
        self.rating_synonyms = {
            'tv-14': ['14 plus', 'TV 14', 'teen rated', 'teenage appropriate'],
            'tv-ma': ['mature', 'adult', 'mature content', 'MA rated'],
            'pg-13': ['teen appropriate', 'family with teens']
        }
        
        # Temporal patterns
        self.temporal_patterns = {
            'recent': ['2020', '2021', '2022', '2023'],
            'modern': ['2010', '2015', '2020'],
            '90s': ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999'],
            '2000s': ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
        }
        
        self.indexed = False
    
    def index_data(self, csv_path: str):
        """Index data across search systems"""
        self.primary_search.index_data(csv_path)
        self.adaptive_search.index_data(csv_path=csv_path)
        
        # Prepare cross-encoder reranking
        prepare_cross_encoder_reranking(csv_path)
        
        self.indexed = True
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Advanced query intent analysis"""
        query_lower = query.lower()
        intent = {
            'type': 'general',
            'actors': [],
            'genres': [],
            'moods': [],
            'temporal': [],
            'themes': [],
            'modifiers': []
        }
        
        # Extract actors (capitalized names)
        actor_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        actors = re.findall(actor_pattern, query)
        intent['actors'] = actors
        if actors:
            intent['type'] = 'actor_based'
        
        # Extract explicit genres
        for genre in self.genre_synonyms:
            if genre in query_lower:
                intent['genres'].append(genre)
                intent['type'] = 'genre_based'
        
        # Extract mood indicators
        for mood, genres in self.mood_to_genre.items():
            if mood in query_lower:
                intent['moods'].append(mood)
                intent['genres'].extend(genres)
                intent['type'] = 'mood_based'
        
        # Extract temporal indicators
        year_matches = re.findall(r'\b(19|20)\d{2}(?:s)?\b', query)
        for match in year_matches:
            if isinstance(match, tuple):
                year = match[0] + match[1]
            else:
                year = match
            intent['temporal'].append(year)
            intent['type'] = 'temporal_based'
        
        # Extract thematic indicators
        theme_keywords = ['about', 'story', 'set in', 'featuring', 'exploring']
        for keyword in theme_keywords:
            if keyword in query_lower:
                intent['type'] = 'theme_based'
                break
        
        return intent
    
    def generate_strategic_queries(self, query: str, intent: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Generate multiple strategic query variations with weights"""
        queries = [(query, 1.0)]  # Original query with highest weight
        
        # Strategy 1: Genre expansion
        if intent['genres']:
            for genre in intent['genres']:
                if genre in self.genre_synonyms:
                    for synonym in self.genre_synonyms[genre][:2]:
                        expanded = query.lower().replace(genre, synonym)
                        if expanded != query.lower():
                            queries.append((expanded, 0.8))
        
        # Strategy 2: Actor name formatting
        if intent['actors']:
            for actor in intent['actors']:
                # "movies with [actor]"
                queries.append((f"movies with {actor}", 0.9))
                queries.append((f"films starring {actor}", 0.9))
                # Just the actor name
                queries.append((actor, 0.7))
        
        # Strategy 3: Mood to genre translation
        if intent['moods']:
            for mood in intent['moods']:
                if mood in self.mood_to_genre:
                    for genre in self.mood_to_genre[mood]:
                        mood_query = query.lower().replace(mood, genre)
                        if mood_query != query.lower():
                            queries.append((mood_query, 0.8))
        
        # Strategy 4: Temporal expansion
        if intent['temporal']:
            for temporal in intent['temporal']:
                # Add temporal context
                queries.append((f"movies from {temporal}", 0.8))
                queries.append((f"released in {temporal}", 0.8))
        
        # Strategy 5: Thematic keyword removal (simplification)
        simplified = query.lower()
        for removal_word in ['about', 'featuring', 'set in', 'with themes of']:
            simplified = simplified.replace(removal_word, '').strip()
        if simplified != query.lower() and len(simplified) > 3:
            queries.append((simplified, 0.6))
        
        # Remove duplicates and limit
        seen = set()
        unique_queries = []
        for q, w in queries:
            q_clean = q.strip().lower()
            if q_clean not in seen and len(q_clean) > 2:
                seen.add(q_clean)
                unique_queries.append((q, w))
        
        return unique_queries[:8]  # Limit to top 8 strategic queries
    
    def execute_multi_strategy_search(self, query_variants: List[Tuple[str, float]], 
                                    config: StrategicSearchConfig) -> Dict[str, Tuple[SearchResult, List[float]]]:
        """Execute search across multiple strategies and variants"""
        all_results = defaultdict(lambda: (None, []))
        
        for query_text, query_weight in query_variants:
            # Strategy 1: Enhanced basic search
            basic_config = SearchConfig(
                boost_weights={
                    'title': 6.0,           # Boost title matches heavily
                    'cast': 5.0,            # Actor searches are important
                    'description': 3.0,      # Thematic content
                    'listed_in': 4.0,       # Genre matching
                    'director': 3.0
                },
                max_results=config.max_results_per_strategy
            )
            
            basic_results = self.primary_search.search(query_text, basic_config)
            
            # Score and store basic results
            for i, result in enumerate(basic_results):
                position_score = (config.max_results_per_strategy - i) / config.max_results_per_strategy
                final_score = position_score * query_weight * 0.4  # 40% weight for basic
                
                if result.id not in all_results or all_results[result.id][0] is None:
                    all_results[result.id] = (result, [])
                all_results[result.id][1].append(('basic', final_score))
            
            # Strategy 2: Adaptive search
            try:
                adaptive_result = self.adaptive_search.retrieve(query_text, top_k=config.max_results_per_strategy)
                adaptive_hits = adaptive_result.get('hits', [])
                
                for i, hit in enumerate(adaptive_hits):
                    source = hit.get('_source', {})
                    doc_id = source.get('show_id')
                    if doc_id:
                        position_score = (config.max_results_per_strategy - i) / config.max_results_per_strategy
                        final_score = position_score * query_weight * 0.6  # 60% weight for adaptive
                        
                        if doc_id not in all_results or all_results[doc_id][0] is None:
                            # Create SearchResult from adaptive hit
                            result = SearchResult(
                                id=doc_id,
                                title=source.get('title', ''),
                                score=final_score,
                                content_type=source.get('type', ''),
                                metadata=source
                            )
                            all_results[doc_id] = (result, [])
                        
                        all_results[doc_id][1].append(('adaptive', final_score))
            except Exception as e:
                # Adaptive search failed, continue with basic only
                pass
        
        return dict(all_results)
    
    def apply_strategic_ranking(self, results: Dict[str, Tuple[SearchResult, List[float]]], 
                              query: str, intent: Dict[str, Any],
                              config: StrategicSearchConfig) -> List[SearchResult]:
        """Apply strategic ranking based on query intent and multi-strategy scores"""
        scored_results = []
        
        for doc_id, (result, scores) in results.items():
            if result is None:
                continue
            
            # Calculate base combined score
            strategy_scores = defaultdict(list)
            for strategy, score in scores:
                strategy_scores[strategy].append(score)
            
            # Average scores per strategy
            combined_score = 0.0
            for strategy, score_list in strategy_scores.items():
                avg_score = sum(score_list) / len(score_list)
                combined_score += avg_score
            
            # Apply strategic boosts
            
            # Boost 1: Exact phrase matches in title
            title = result.title.lower()
            query_lower = query.lower()
            if query_lower in title or title in query_lower:
                combined_score *= config.boost_exact_phrase_match
            
            # Boost 2: Actor name matches
            if intent['actors']:
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                cast = metadata.get('cast', '').lower() if metadata else ''
                for actor in intent['actors']:
                    if actor.lower() in cast:
                        combined_score *= 2.5  # Strong boost for correct actor
            
            # Boost 3: Multi-field matches
            fields_matched = 0
            if hasattr(result, 'metadata') and result.metadata:
                metadata = result.metadata
                for field in ['title', 'description', 'listed_in', 'cast']:
                    field_value = str(metadata.get(field, '')).lower()
                    if any(word in field_value for word in query_lower.split() if len(word) > 2):
                        fields_matched += 1
                
                if fields_matched >= 2:
                    combined_score *= config.boost_multi_field_match
            
            # Boost 4: Genre/mood alignment
            if intent['genres'] and hasattr(result, 'metadata') and result.metadata:
                genres = str(result.metadata.get('listed_in', '')).lower()
                for expected_genre in intent['genres']:
                    if expected_genre in genres:
                        combined_score *= 1.8
            
            # Final score assignment
            result.score = combined_score
            scored_results.append(result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply cross-encoder reranking for semantic boost
        if config.use_cross_encoder and scored_results:
            reranker = get_cross_encoder_reranker()
            # Rerank top candidates with cross-encoder
            top_candidates = scored_results[:min(80, len(scored_results))]
            scored_results = reranker.rerank(
                query, 
                top_candidates, 
                top_k=config.final_result_count,
                semantic_weight=config.cross_encoder_weight
            )
        
        return scored_results[:config.final_result_count]
    
    def strategic_search(self, query: str, config: Optional[StrategicSearchConfig] = None) -> List[SearchResult]:
        """Main strategic search method"""
        if not self.indexed:
            raise RuntimeError("Data not indexed. Call index_data() first.")
        
        if not config:
            config = StrategicSearchConfig()
        
        # Step 1: Analyze query intent
        intent = self.analyze_query_intent(query)
        
        # Step 2: Generate strategic query variations
        query_variants = self.generate_strategic_queries(query, intent)
        
        # Step 3: Execute multi-strategy search
        raw_results = self.execute_multi_strategy_search(query_variants, config)
        
        # Step 4: Apply strategic ranking
        final_results = self.apply_strategic_ranking(raw_results, query, intent, config)
        
        return final_results
    
    def search(self, query: str, config: Optional[StrategicSearchConfig] = None) -> List[SearchResult]:
        """Main search interface (compatible with evaluator)"""
        return self.strategic_search(query, config)