"""
Intelligent Tag Enrichment System

Automatically extracts and adds semantic tags to improve search performance:
- Actor name normalization and aliases
- Director style and era tags  
- Genre expansion and synonyms
- Theme extraction from descriptions
- Decade and era categorization
- Content mood and tone analysis
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentTagger:
    """
    Intelligent tagging system that enriches movie/TV data with searchable tags
    """
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.embedder = None
        self.genre_themes = {}
        self.director_styles = {}
        self.actor_aliases = {}
        
        self._init_models()
        self._init_knowledge_base()
    
    def _init_models(self):
        """Initialize NLP models for tag extraction"""
        try:
            logger.info("Loading NLP models...")
            
            # SpaCy for NER and text processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Sentiment analysis for mood tags
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            except Exception:
                logger.warning("Sentiment analyzer not available")
                self.sentiment_analyzer = None
            
            logger.info("✅ Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _init_knowledge_base(self):
        """Initialize knowledge base for intelligent tagging"""
        
        # Genre theme mapping for semantic expansion
        self.genre_themes = {
            'action': ['explosive', 'adrenaline', 'intense', 'thrilling', 'fast-paced', 'combat', 'adventure'],
            'comedy': ['funny', 'hilarious', 'humor', 'laugh', 'witty', 'satirical', 'amusing'],
            'drama': ['emotional', 'character-driven', 'serious', 'compelling', 'dramatic', 'moving'],
            'horror': ['scary', 'frightening', 'suspenseful', 'creepy', 'supernatural', 'terrifying'],
            'romance': ['romantic', 'love', 'relationship', 'heartwarming', 'passionate', 'intimate'],
            'sci-fi': ['futuristic', 'technology', 'space', 'aliens', 'scientific', 'dystopian'],
            'thriller': ['suspenseful', 'edge-of-seat', 'mystery', 'tense', 'psychological'],
            'documentary': ['educational', 'informative', 'real-life', 'factual', 'investigative']
        }
        
        # Era and decade tags
        self.era_mapping = {
            (1920, 1929): ['1920s', 'silent-era', 'roaring-twenties'],
            (1930, 1939): ['1930s', 'great-depression', 'golden-age-hollywood'],
            (1940, 1949): ['1940s', 'world-war-2', 'film-noir-era'],
            (1950, 1959): ['1950s', 'post-war', 'classic-hollywood'],
            (1960, 1969): ['1960s', 'new-hollywood', 'counterculture'],
            (1970, 1979): ['1970s', 'new-hollywood', 'auteur-cinema'],
            (1980, 1989): ['1980s', 'blockbuster-era', 'mtv-generation'],
            (1990, 1999): ['1990s', 'independent-film', 'pre-digital'],
            (2000, 2009): ['2000s', 'digital-era', 'early-streaming'],
            (2010, 2019): ['2010s', 'streaming-era', 'superhero-boom'],
            (2020, 2029): ['2020s', 'pandemic-era', 'streaming-native']
        }
        
        # Common director style associations (learned from data)
        self.director_style_keywords = [
            'auteur', 'visionary', 'innovative', 'stylistic', 'signature',
            'acclaimed', 'award-winning', 'masterpiece', 'cult-classic'
        ]
    
    def extract_actor_tags(self, cast_str: str) -> Dict[str, Any]:
        """Extract and normalize actor information with aliases"""
        if pd.isna(cast_str) or not cast_str:
            return {'actors': [], 'lead_actors': [], 'actor_count': 0}
        
        # Split and clean actor names
        actors = [name.strip() for name in str(cast_str).split(',') if name.strip()]
        
        # Get lead actors (usually first 3)
        lead_actors = actors[:3]
        
        # Generate searchable variations
        actor_variations = []
        for actor in actors:
            actor_variations.append(actor)
            # Add first name + last name variations
            parts = actor.split()
            if len(parts) >= 2:
                actor_variations.append(f"{parts[0]} {parts[-1]}")  # First + Last name
                actor_variations.append(parts[-1])  # Last name only
        
        return {
            'actors': actors,
            'lead_actors': lead_actors,
            'actor_count': len(actors),
            'actor_variations': list(set(actor_variations))
        }
    
    def extract_director_tags(self, director_str: str) -> Dict[str, Any]:
        """Extract director information with style indicators"""
        if pd.isna(director_str) or not director_str:
            return {'directors': [], 'director_count': 0}
        
        # Split and clean director names
        directors = [name.strip() for name in str(director_str).split(',') if name.strip()]
        
        # Generate searchable variations
        director_variations = []
        for director in directors:
            director_variations.append(director)
            # Add first name + last name variations
            parts = director.split()
            if len(parts) >= 2:
                director_variations.append(f"{parts[0]} {parts[-1]}")
                director_variations.append(parts[-1])  # Last name only for queries like "Nolan movies"
        
        return {
            'directors': directors,
            'director_count': len(directors),
            'director_variations': list(set(director_variations))
        }
    
    def extract_genre_tags(self, genre_str: str) -> Dict[str, Any]:
        """Extract and expand genre information with semantic tags"""
        if pd.isna(genre_str) or not genre_str:
            return {'genres': [], 'genre_themes': [], 'primary_genre': None}
        
        # Clean and split genres
        genres = [genre.strip() for genre in str(genre_str).split(',') if genre.strip()]
        
        # Extract theme tags based on genres
        theme_tags = []
        for genre in genres:
            genre_lower = genre.lower()
            for base_genre, themes in self.genre_themes.items():
                if base_genre in genre_lower or any(keyword in genre_lower for keyword in [base_genre]):
                    theme_tags.extend(themes)
        
        # Determine primary genre (first one usually most important)
        primary_genre = genres[0] if genres else None
        
        return {
            'genres': genres,
            'genre_themes': list(set(theme_tags)),
            'primary_genre': primary_genre,
            'genre_count': len(genres)
        }
    
    def extract_description_tags(self, description: str) -> Dict[str, Any]:
        """Extract semantic tags from description using NLP"""
        if pd.isna(description) or not description:
            return {'themes': [], 'mood': 'neutral', 'entities': []}
        
        desc_str = str(description)
        result = {
            'themes': [],
            'mood': 'neutral',
            'entities': [],
            'keywords': []
        }
        
        # Extract mood using sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment_results = self.sentiment_analyzer(desc_str[:500])
                if sentiment_results:
                    # Get dominant sentiment
                    top_sentiment = max(sentiment_results[0], key=lambda x: x['score'])
                    if top_sentiment['score'] > 0.6:
                        if top_sentiment['label'] == 'LABEL_0':  # Negative
                            result['mood'] = 'dark'
                        elif top_sentiment['label'] == 'LABEL_2':  # Positive
                            result['mood'] = 'uplifting'
                        else:
                            result['mood'] = 'neutral'
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
        
        # Extract entities using SpaCy
        if self.nlp:
            try:
                doc = self.nlp(desc_str)
                entities = []
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                        entities.append(ent.text)
                result['entities'] = entities[:10]  # Limit entities
                
                # Extract key themes from description
                theme_keywords = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'ADJ'] and 
                        len(token.text) > 3 and 
                        not token.is_stop and 
                        token.is_alpha):
                        theme_keywords.append(token.lemma_.lower())
                
                # Filter to meaningful themes
                result['keywords'] = list(set(theme_keywords))[:15]
                
            except Exception as e:
                logger.debug(f"NLP processing failed: {e}")
        
        # Extract manual themes using keyword matching
        theme_patterns = {
            'family': ['family', 'father', 'mother', 'son', 'daughter', 'parent', 'child'],
            'friendship': ['friend', 'friendship', 'buddy', 'companion'],
            'revenge': ['revenge', 'vengeance', 'payback', 'retribution'],
            'love': ['love', 'romance', 'relationship', 'marry', 'wedding'],
            'war': ['war', 'battle', 'conflict', 'soldier', 'military'],
            'crime': ['crime', 'criminal', 'murder', 'detective', 'investigation'],
            'supernatural': ['ghost', 'spirit', 'supernatural', 'magic', 'paranormal'],
            'coming-of-age': ['young', 'teenager', 'adolescent', 'growing up'],
            'survival': ['survive', 'survival', 'escape', 'trapped', 'rescue']
        }
        
        desc_lower = desc_str.lower()
        themes = []
        for theme, keywords in theme_patterns.items():
            if any(keyword in desc_lower for keyword in keywords):
                themes.append(theme)
        
        result['themes'] = themes
        return result
    
    def extract_temporal_tags(self, release_year: Any) -> Dict[str, Any]:
        """Extract decade and era tags"""
        result = {
            'decade': None,
            'era_tags': [],
            'year_category': None
        }
        
        try:
            year = int(release_year) if pd.notna(release_year) else None
            if year:
                # Decade
                decade = (year // 10) * 10
                result['decade'] = f"{decade}s"
                
                # Era tags
                for (start, end), tags in self.era_mapping.items():
                    if start <= year <= end:
                        result['era_tags'] = tags
                        break
                
                # Category based on recency
                current_year = 2024
                if year >= current_year - 3:
                    result['year_category'] = 'recent'
                elif year >= current_year - 10:
                    result['year_category'] = 'modern'
                elif year >= current_year - 30:
                    result['year_category'] = 'classic-modern'
                else:
                    result['year_category'] = 'classic'
                    
        except (ValueError, TypeError):
            pass
        
        return result
    
    def extract_content_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract content type and duration metadata"""
        result = {
            'content_type': row.get('type', '').lower(),
            'duration_category': None,
            'rating_category': None
        }
        
        # Duration categorization
        duration = str(row.get('duration', ''))
        if 'min' in duration:
            try:
                minutes = int(re.search(r'(\d+)', duration).group(1))
                if minutes <= 90:
                    result['duration_category'] = 'short'
                elif minutes <= 120:
                    result['duration_category'] = 'standard'
                else:
                    result['duration_category'] = 'long'
            except:
                pass
        elif 'Season' in duration:
            result['duration_category'] = 'series'
        
        # Rating categorization
        rating = str(row.get('rating', ''))
        if rating in ['G', 'PG', 'PG-13']:
            result['rating_category'] = 'family-friendly'
        elif rating in ['R', 'TV-MA']:
            result['rating_category'] = 'mature'
        else:
            result['rating_category'] = 'general'
        
        return result
    
    def enrich_single_row(self, row: pd.Series) -> Dict[str, Any]:
        """Enrich a single row with comprehensive tags"""
        
        # Extract all tag categories
        actor_tags = self.extract_actor_tags(row.get('cast', ''))
        director_tags = self.extract_director_tags(row.get('director', ''))
        genre_tags = self.extract_genre_tags(row.get('listed_in', ''))
        description_tags = self.extract_description_tags(row.get('description', ''))
        temporal_tags = self.extract_temporal_tags(row.get('release_year'))
        content_tags = self.extract_content_metadata(row)
        
        # Combine all tags into comprehensive tag string
        all_tags = []
        
        # Add actor tags
        if actor_tags['actor_variations']:
            all_tags.extend(actor_tags['actor_variations'][:10])  # Limit to avoid bloat
        
        # Add director tags  
        if director_tags['director_variations']:
            all_tags.extend(director_tags['director_variations'])
        
        # Add genre and theme tags
        if genre_tags['genres']:
            all_tags.extend(genre_tags['genres'])
        if genre_tags['genre_themes']:
            all_tags.extend(genre_tags['genre_themes'][:5])
        
        # Add description themes
        if description_tags['themes']:
            all_tags.extend(description_tags['themes'])
        if description_tags['keywords']:
            all_tags.extend(description_tags['keywords'][:8])
        
        # Add temporal tags
        if temporal_tags['decade']:
            all_tags.append(temporal_tags['decade'])
        if temporal_tags['era_tags']:
            all_tags.extend(temporal_tags['era_tags'][:3])
        if temporal_tags['year_category']:
            all_tags.append(temporal_tags['year_category'])
        
        # Add content metadata
        if content_tags['duration_category']:
            all_tags.append(content_tags['duration_category'])
        if content_tags['rating_category']:
            all_tags.append(content_tags['rating_category'])
        
        # Add mood
        if description_tags['mood'] != 'neutral':
            all_tags.append(description_tags['mood'])
        
        return {
            'semantic_tags': ' | '.join(list(set(all_tags))),  # Deduplicated tags
            'actor_count': actor_tags['actor_count'],
            'director_count': director_tags['director_count'],
            'genre_count': genre_tags['genre_count'],
            'primary_genre': genre_tags['primary_genre'],
            'mood': description_tags['mood'],
            'decade': temporal_tags['decade'],
            'year_category': temporal_tags['year_category'],
            'content_type': content_tags['content_type'],
            'duration_category': content_tags['duration_category'],
            'rating_category': content_tags['rating_category']
        }
    
    def enrich_dataset(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        """Enrich entire dataset with intelligent tags"""
        
        logger.info(f"Loading dataset from {input_csv}")
        df = pd.read_csv(input_csv, encoding='latin-1').fillna('')
        
        logger.info(f"Processing {len(df)} rows...")
        
        # Initialize new columns
        enrichment_columns = [
            'semantic_tags', 'actor_count', 'director_count', 'genre_count',
            'primary_genre', 'mood', 'decade', 'year_category', 'content_type',
            'duration_category', 'rating_category'
        ]
        
        for col in enrichment_columns:
            df[col] = ''
        
        # Process each row
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} rows...")
            
            try:
                enrichment = self.enrich_single_row(row)
                
                # Add enrichment data to dataframe
                for key, value in enrichment.items():
                    df.at[idx, key] = value
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        # Save enriched dataset
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"✅ Enriched dataset saved to {output_csv}")
        
        # Print enrichment statistics
        logger.info("Enrichment Statistics:")
        logger.info(f"- Total rows processed: {len(df)}")
        logger.info(f"- Unique primary genres: {df['primary_genre'].nunique()}")
        logger.info(f"- Average tags per row: {df['semantic_tags'].str.split('|').str.len().mean():.1f}")
        logger.info(f"- Decade distribution: {df['decade'].value_counts().head()}")
        
        return df


def main():
    """Main function to run enrichment"""
    tagger = IntelligentTagger()
    
    input_file = "data/netflix_titles_cleaned.csv"
    output_file = "data/netflix_titles_enriched.csv"
    
    enriched_df = tagger.enrich_dataset(input_file, output_file)
    
    print("\n🎯 Sample enriched row:")
    sample_row = enriched_df.iloc[0]
    print(f"Title: {sample_row['title']}")
    print(f"Semantic Tags: {sample_row['semantic_tags']}")
    print(f"Primary Genre: {sample_row['primary_genre']}")
    print(f"Mood: {sample_row['mood']}")
    print(f"Decade: {sample_row['decade']}")


if __name__ == "__main__":
    main()