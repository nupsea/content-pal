"""
Query classification for adaptive retrieval
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Query intent types"""
    TITLE_SEARCH = "title_search"
    ACTOR_SEARCH = "actor_search"
    GENRE_SEARCH = "genre_search"
    YEAR_SEARCH = "year_search"
    DIRECTOR_SEARCH = "director_search"
    CONTENT_SEARCH = "content_search"
    COMBINED_SEARCH = "combined_search"
    UNKNOWN = "unknown"


@dataclass
class QueryIntent:
    """Query intent classification result"""
    intent_type: IntentType
    confidence: float
    extracted_entities: Dict[str, List[str]]
    reasoning: str


class QueryClassifier:
    """Classifies search queries to determine optimal retrieval strategy"""
    
    def __init__(self):
        # Actor name patterns
        self.actor_indicators = {
            'movies with', 'shows with', 'films with', 'starring',
            'actor', 'actress', 'cast'
        }
        
        # Genre terms
        self.genre_terms = {
            'action', 'comedy', 'drama', 'horror', 'thriller', 'romance', 'sci-fi',
            'fantasy', 'documentary', 'animation', 'adventure', 'crime', 'mystery',
            'musical', 'western', 'war', 'family', 'kids', 'children', 'reality',
            'series', 'show', 'movie', 'film'
        }
        
        # Director indicators
        self.director_indicators = {
            'directed by', 'director', 'filmmaker', 'made by'
        }
        
        # Year patterns
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        self.year_indicators = {
            'from', 'in', 'released', 'year', 'recent', 'new', 'latest', 'old'
        }
        
        # Content/theme indicators
        self.content_indicators = {
            'about', 'story', 'plot', 'character', 'based on', 'featuring'
        }
    
    def classify_query(self, query: str) -> QueryIntent:
        """Classify query intent"""
        query_lower = query.lower().strip()
        tokens = query_lower.split()
        
        # Initialize scores
        scores = {intent: 0.0 for intent in IntentType}
        entities = {
            'actors': [],
            'genres': [],
            'years': [],
            'directors': [],
            'titles': [],
            'themes': []
        }
        
        # Check for year patterns
        years = self.year_pattern.findall(query)
        if years:
            entities['years'] = years
            scores[IntentType.YEAR_SEARCH] += 0.8
            
            # Boost if year indicators present
            if any(indicator in query_lower for indicator in self.year_indicators):
                scores[IntentType.YEAR_SEARCH] += 0.2
        
        # Check for actor indicators
        for indicator in self.actor_indicators:
            if indicator in query_lower:
                scores[IntentType.ACTOR_SEARCH] += 0.7
                # Extract potential actor name after indicator
                if indicator in ['movies with', 'shows with', 'films with']:
                    parts = query_lower.split(indicator, 1)
                    if len(parts) > 1:
                        potential_actor = parts[1].strip()
                        entities['actors'].append(potential_actor)
        
        # Check for actor name patterns (First Last + movies/films/shows)
        import re
        actor_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(movies?|films?|shows?)'
        matches = re.findall(actor_pattern, query)
        if matches:
            scores[IntentType.ACTOR_SEARCH] += 0.9
            for match in matches:
                entities['actors'].append(match[0])
        
        # Check for common actor name patterns in general
        # Names with capital letters followed by media terms
        name_media_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s+(movies?|films?|shows?))?'
        name_matches = re.findall(name_media_pattern, query)
        if name_matches and len(tokens) <= 4:  # Short queries with names likely actor searches
            scores[IntentType.ACTOR_SEARCH] += 0.6
            for match in name_matches:
                entities['actors'].append(match[0])
        
        # Check for genre terms
        genre_matches = [term for term in tokens if term in self.genre_terms]
        if genre_matches:
            entities['genres'] = genre_matches
            scores[IntentType.GENRE_SEARCH] += len(genre_matches) * 0.4
        
        # Check for director indicators
        for indicator in self.director_indicators:
            if indicator in query_lower:
                scores[IntentType.DIRECTOR_SEARCH] += 0.7
        
        # Check for content/theme indicators
        content_matches = [term for term in self.content_indicators if term in query_lower]
        if content_matches:
            entities['themes'] = content_matches
            scores[IntentType.CONTENT_SEARCH] += len(content_matches) * 0.3
        
        # Title search heuristics
        # Short queries without other indicators likely title searches
        if len(tokens) <= 3 and not any(scores[intent] > 0 for intent in 
                                       [IntentType.ACTOR_SEARCH, IntentType.GENRE_SEARCH, 
                                        IntentType.YEAR_SEARCH, IntentType.DIRECTOR_SEARCH]):
            scores[IntentType.TITLE_SEARCH] += 0.6
        
        # Check for quoted phrases (likely titles)
        if '"' in query or "'" in query:
            scores[IntentType.TITLE_SEARCH] += 0.5
        
        # Proper nouns (capitalized words) might indicate titles or names
        proper_nouns = [token for token in query.split() if token[0].isupper()]
        if proper_nouns and len(proper_nouns) >= 2:
            scores[IntentType.TITLE_SEARCH] += 0.3
            entities['titles'] = proper_nouns
        
        # Combined search if multiple high scores
        high_scores = [intent for intent, score in scores.items() if score >= 0.5]
        if len(high_scores) > 1:
            scores[IntentType.COMBINED_SEARCH] = max(scores.values()) + 0.2
        
        # Find best intent
        best_intent = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_intent]
        
        # Default to unknown if score too low
        if best_score < 0.3:
            best_intent = IntentType.UNKNOWN
            best_score = 0.1
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_intent, entities, best_score)
        
        return QueryIntent(
            intent_type=best_intent,
            confidence=min(best_score, 1.0),
            extracted_entities=entities,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, intent: IntentType, entities: Dict, confidence: float) -> str:
        """Generate human-readable reasoning for classification"""
        if intent == IntentType.TITLE_SEARCH:
            return f"Appears to be a title search (confidence: {confidence:.2f})"
        elif intent == IntentType.ACTOR_SEARCH:
            actors = entities.get('actors', [])
            actor_text = f" for {', '.join(actors[:2])}" if actors else ""
            return f"Actor/cast search{actor_text} (confidence: {confidence:.2f})"
        elif intent == IntentType.GENRE_SEARCH:
            genres = entities.get('genres', [])
            genre_text = f" for {', '.join(genres[:2])}" if genres else ""
            return f"Genre search{genre_text} (confidence: {confidence:.2f})"
        elif intent == IntentType.YEAR_SEARCH:
            years = entities.get('years', [])
            year_text = f" for {', '.join(years)}" if years else ""
            return f"Year-based search{year_text} (confidence: {confidence:.2f})"
        elif intent == IntentType.DIRECTOR_SEARCH:
            return f"Director search (confidence: {confidence:.2f})"
        elif intent == IntentType.CONTENT_SEARCH:
            return f"Content/theme search (confidence: {confidence:.2f})"
        elif intent == IntentType.COMBINED_SEARCH:
            return f"Combined search strategy (confidence: {confidence:.2f})"
        else:
            return f"Unknown intent (confidence: {confidence:.2f})"