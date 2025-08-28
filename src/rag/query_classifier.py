#!/usr/bin/env python3
"""
ML-based query classifier for Netflix content recommendations.
Uses pre-trained models for intent classification and named entity recognition.
"""

import re
import spacy
from typing import Dict, List, Optional
from functools import lru_cache
from dataclasses import dataclass

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class QueryIntent:
    """Structured representation of query analysis"""
    intent_type: str  # TITLE, ACTOR, GENRE, YEAR, DIRECTOR, CONTENT, HYBRID
    confidence: float  # 0.0 to 1.0
    entities: Dict[str, List[str]]  # extracted entities by type
    search_strategy: str  # keyword, semantic, hybrid, filtered
    filters: Dict[str, str]  # additional filters to apply
    explanation: str  # reasoning for classification


class QueryClassifier:
    """
    ML-based query classifier using pre-trained models.
    Much simpler and more accurate than rule-based approaches.
    """
    
    # Intent categories for zero-shot classification
    INTENT_LABELS = [
        "title search",
        "actor search", 
        "genre search",
        "year search",
        "director search",
        "content theme search",
        "hybrid search"
    ]
    
    # Mapping from classifier labels to our intent types
    LABEL_MAPPING = {
        "title search": "TITLE",
        "actor search": "ACTOR", 
        "genre search": "GENRE",
        "year search": "YEAR",
        "director search": "DIRECTOR",
        "content theme search": "CONTENT",
        "hybrid search": "HYBRID"
    }
    
    # Common genre terms for validation
    GENRE_TERMS = {
        'action', 'comedy', 'drama', 'horror', 'thriller', 'romance', 
        'sci-fi', 'fantasy', 'documentary', 'animation', 'adventure', 
        'crime', 'mystery', 'musical', 'western', 'war', 'family', 
        'kids', 'children', 'reality'
    }

    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml and HAS_TRANSFORMERS
        self.nlp = None
        self.classifier = None
        
        if self.use_ml:
            try:
                # Load spacy model for NER
                self.nlp = spacy.load("en_core_web_sm")
                
                # Load zero-shot classifier with a smaller, faster model
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=-1  # Use CPU for better compatibility
                )
                
            except Exception as e:
                print(f"Failed to load ML models: {e}")
                print("Falling back to rule-based classification")
                self.use_ml = False
    
    def classify(self, query: str) -> QueryIntent:
        """Main classification method"""
        query = query.strip()
        
        if self.use_ml:
            return self._classify_with_ml(query)
        else:
            return self._classify_with_rules(query)
    
    def _classify_with_ml(self, query: str) -> QueryIntent:
        """ML-based classification using zero-shot classification and NER"""
        
        # Get intent classification
        result = self.classifier(query, self.INTENT_LABELS)
        intent_label = result['labels'][0]
        confidence = result['scores'][0]
        
        # Map to our intent types
        intent_type = self.LABEL_MAPPING.get(intent_label, "HYBRID")
        
        # Extract entities using spaCy
        doc = self.nlp(query)
        entities = {
            "actors": [],
            "directors": [],
            "genres": [],
            "years": [],
            "titles": [],
            "themes": []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Assume persons are actors unless context suggests director
                if "director" in query.lower() or "directed by" in query.lower():
                    entities["directors"].append(ent.text)
                else:
                    entities["actors"].append(ent.text)
            elif ent.label_ == "DATE":
                # Extract years from dates
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', ent.text)
                if year_match:
                    entities["years"].append(year_match.group())
            elif ent.label_ in ["WORK_OF_ART", "EVENT"]:
                entities["titles"].append(ent.text)
        
        # Extract genres using token matching
        tokens = [token.text.lower() for token in doc]
        for token in tokens:
            if token in self.GENRE_TERMS:
                entities["genres"].append(token)
        
        # Extract years using regex as backup
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        entities["years"].extend(year_matches)
        
        # Extract theme elements for content searches
        if intent_type == "CONTENT":
            # Use noun phrases as theme elements
            entities["themes"] = [chunk.text for chunk in doc.noun_chunks]
        
        # Clean and deduplicate entities
        entities = {k: list(set(v)) for k, v in entities.items() if v}
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(intent_type, entities)
        
        # Create filters
        filters = self._create_filters(entities)
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            search_strategy=search_strategy,
            filters=filters,
            explanation=f"ML-based: {intent_label} (confidence: {confidence:.2f})"
        )
    
    def _classify_with_rules(self, query: str) -> QueryIntent:
        """Improved rule-based classifier optimized for Netflix content search"""
        query_lower = query.lower()
        entities = {"actors": [], "genres": [], "years": [], "directors": [], "themes": []}
        intent_type = "HYBRID"  # Default
        confidence = 0.5
        
        # Actor patterns - high priority
        # Pattern 1: "Name movies/films/shows"
        name_media_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(movies?|films?|shows?)\b'
        matches = re.findall(name_media_pattern, query)
        if matches:
            intent_type = "ACTOR"
            confidence = 0.9
            for name, media_type in matches:
                entities["actors"].append(name)
        
        # Pattern 2: "starring/featuring/with Name"
        elif re.search(r'\b(with|starring|featuring)\s+[A-Z]', query):
            starring_match = re.search(r'\b(?:with|starring|featuring)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
            if starring_match:
                intent_type = "ACTOR"  
                confidence = 0.9
                entities["actors"].append(starring_match.group(1))
        
        # Director patterns
        if "directed by" in query_lower or "director" in query_lower:
            director_match = re.search(r'directed by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
            if director_match:
                entities["directors"].append(director_match.group(1))
                intent_type = "DIRECTOR"
                confidence = 0.9
        
        # Genre detection - more comprehensive
        found_genres = []
        for genre in self.GENRE_TERMS:
            if genre in query_lower:
                found_genres.append(genre)
        
        if found_genres and intent_type == "HYBRID":
            entities["genres"] = found_genres
            intent_type = "GENRE"
            confidence = 0.8
        
        # Year patterns
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        if years:
            entities["years"] = years
            if intent_type == "HYBRID":
                intent_type = "YEAR"
                confidence = 0.8
        
        # Title search indicators
        if intent_type == "HYBRID":
            # Short queries are likely titles
            if len(query.split()) <= 2:
                intent_type = "TITLE"
                confidence = 0.7
            # Quoted phrases are titles
            elif '"' in query or "'" in query:
                intent_type = "TITLE"
                confidence = 0.9
            # Proper nouns might be titles
            elif any(word[0].isupper() for word in query.split()):
                intent_type = "TITLE"
                confidence = 0.6
            # Descriptive phrases are content searches
            elif len(query.split()) > 5:
                intent_type = "CONTENT"
                confidence = 0.7
                entities["themes"] = query.split()[:8]
        
        # Boost confidence if multiple signals
        signal_count = sum([
            bool(entities.get("actors")),
            bool(entities.get("genres")), 
            bool(entities.get("years")),
            bool(entities.get("directors"))
        ])
        
        if signal_count > 1:
            confidence = min(confidence + 0.1, 1.0)
        
        search_strategy = self._determine_search_strategy(intent_type, entities)
        filters = self._create_filters(entities)
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            search_strategy=search_strategy,
            filters=filters,
            explanation=f"Rule-based: {intent_type.lower()} intent"
        )
    
    def _determine_search_strategy(self, intent_type: str, entities: Dict[str, List[str]]) -> str:
        """Determine optimal search strategy based on intent and entities"""
        if intent_type in ["ACTOR", "DIRECTOR", "TITLE"]:
            return "keyword"
        elif intent_type == "GENRE":
            return "filtered"
        elif intent_type == "CONTENT":
            return "semantic"
        elif intent_type == "YEAR":
            return "filtered"
        else:
            return "hybrid"
    
    def _create_filters(self, entities: Dict[str, List[str]]) -> Dict[str, str]:
        """Create search filters based on extracted entities"""
        filters = {}
        
        if entities.get("genres"):
            filters["genre"] = entities["genres"][0]
        
        if entities.get("years"):
            year = entities["years"][0]
            filters["year_range"] = f"{year}-{year}"
        
        return filters


# Convenience functions for quick classification
@lru_cache(maxsize=1000)
def classify_query(query: str) -> QueryIntent:
    """Cached query classification"""
    classifier = QueryClassifier()
    return classifier.classify(query)


def batch_classify(queries: List[str]) -> List[QueryIntent]:
    """Classify multiple queries efficiently"""
    classifier = QueryClassifier()
    return [classifier.classify(q) for q in queries]


if __name__ == "__main__":
    # Test the classifier
    test_queries = [
        "Bruce Greenwood movies",
        "documentary about chess", 
        "feel-good romantic comedies",
        "movies from 2023",
        "story about a chess prodigy who overcomes challenges",
        "thriller movies with suspense",
        "movies directed by Christopher Nolan",
        "The Dark Knight"
    ]
    
    print("Testing ML-based Query Classifier:")
    print("=" * 60)
    
    classifier = QueryClassifier()
    for query in test_queries:
        intent = classifier.classify(query)
        print(f"\nQuery: '{query}'")
        print(f"Intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        print(f"Strategy: {intent.search_strategy}")
        if intent.entities:
            print(f"Entities: {intent.entities}")
        if intent.filters:
            print(f"Filters: {intent.filters}")
        print(f"Explanation: {intent.explanation}")