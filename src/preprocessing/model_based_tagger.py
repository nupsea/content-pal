"""
Model-Based Tag Enrichment System

Uses pre-trained NLP models and libraries to automatically extract semantic tags:
- Zero-shot classification for themes and genres
- NER for entity extraction
- Sentence transformers for semantic similarity
- KeyBERT for keyword extraction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import spacy
try:
    from keybert import KeyBERT
except ImportError:
    KeyBERT = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBasedTagger:
    """
    Model-based tagging system using pre-trained NLP models
    """
    
    def __init__(self):
        self.models = {}
        self._init_models()
    
    def _init_models(self):
        """Initialize all NLP models"""
        logger.info("Loading NLP models...")
        
        # 1. Zero-shot classifier for theme/genre classification
        try:
            self.models['zero_shot'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("✅ Zero-shot classifier loaded")
        except Exception as e:
            logger.warning(f"Zero-shot classifier failed: {e}")
            self.models['zero_shot'] = None
        
        # 2. NER model for entity extraction
        try:
            self.models['ner'] = spacy.load("en_core_web_sm")
            logger.info("✅ NER model loaded")
        except Exception as e:
            logger.warning(f"NER model failed: {e}")
            self.models['ner'] = None
        
        # 3. Sentence transformer for semantic similarity
        try:
            self.models['embedder'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"Sentence transformer failed: {e}")
            self.models['embedder'] = None
        
        # 4. KeyBERT for keyword extraction
        if KeyBERT:
            try:
                self.models['keybert'] = KeyBERT(model=self.models['embedder'])
                logger.info("✅ KeyBERT loaded")
            except Exception as e:
                logger.warning(f"KeyBERT failed: {e}")
                self.models['keybert'] = None
        else:
            logger.info("KeyBERT not installed - keyword extraction disabled")
            self.models['keybert'] = None
        
        # 5. Sentiment analyzer for mood detection
        try:
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None
            )
            logger.info("✅ Sentiment analyzer loaded")
        except Exception as e:
            logger.warning(f"Sentiment analyzer failed: {e}")
            self.models['sentiment'] = None
    
    def extract_themes_with_zero_shot(self, description: str) -> List[str]:
        """Use zero-shot classification to detect themes in descriptions"""
        if not self.models['zero_shot'] or pd.isna(description) or not description:
            return []
        
        # Define candidate themes for classification
        candidate_themes = [
            "family drama", "romantic comedy", "action adventure", "psychological thriller",
            "coming of age", "supernatural horror", "crime investigation", "war drama",
            "science fiction", "documentary", "friendship", "revenge story",
            "survival thriller", "musical", "sports drama", "historical fiction",
            "dark comedy", "mystery", "fantasy adventure", "biographical drama"
        ]
        
        try:
            # Truncate description to avoid token limits
            desc_sample = str(description)[:500]
            
            result = self.models['zero_shot'](desc_sample, candidate_themes)
            
            # Get themes with confidence > 0.3
            relevant_themes = []
            for label, score in zip(result['labels'], result['scores']):
                if score > 0.3:  # Confidence threshold
                    relevant_themes.append(label.replace(' ', '-'))
            
            return relevant_themes[:5]  # Top 5 themes
            
        except Exception as e:
            logger.debug(f"Zero-shot classification failed: {e}")
            return []
    
    def extract_entities_with_ner(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy NER"""
        if not self.models['ner'] or pd.isna(text) or not text:
            return {'persons': [], 'orgs': [], 'locations': []}
        
        try:
            doc = self.models['ner'](str(text))
            entities = {
                'persons': [],
                'orgs': [],
                'locations': []
            }
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities['persons'].append(ent.text)
                elif ent.label_ == "ORG":
                    entities['orgs'].append(ent.text)
                elif ent.label_ == "GPE":  # Geopolitical entities (countries, cities)
                    entities['locations'].append(ent.text)
            
            # Deduplicate and limit
            for key in entities:
                entities[key] = list(set(entities[key]))[:5]
            
            return entities
            
        except Exception as e:
            logger.debug(f"NER extraction failed: {e}")
            return {'persons': [], 'orgs': [], 'locations': []}
    
    def extract_keywords_with_keybert(self, text: str) -> List[str]:
        """Extract keywords using KeyBERT"""
        if not self.models['keybert'] or pd.isna(text) or not text:
            return []
        
        try:
            # Extract keywords with diversity for better coverage
            keywords = self.models['keybert'].extract_keywords(
                str(text),
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                use_maxsum=True,
                nr_candidates=20,
                top_k=8
            )
            
            # Return just the keyword strings
            return [kw[0] for kw in keywords]
            
        except Exception as e:
            logger.debug(f"KeyBERT extraction failed: {e}")
            return []
    
    def detect_mood_with_sentiment(self, description: str) -> str:
        """Detect mood using sentiment analysis"""
        if not self.models['sentiment'] or pd.isna(description) or not description:
            return "neutral"
        
        try:
            # Analyze sentiment
            result = self.models['sentiment'](str(description)[:500])
            
            if result and len(result) > 0:
                top_sentiment = max(result, key=lambda x: x['score'])
                
                if top_sentiment['score'] > 0.6:
                    label = top_sentiment['label']
                    if 'NEGATIVE' in label.upper():
                        return "dark"
                    elif 'POSITIVE' in label.upper():
                        return "uplifting"
                    else:
                        return "neutral"
            
            return "neutral"
            
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return "neutral"
    
    def normalize_names_with_ner(self, names_str: str) -> List[str]:
        """Normalize actor/director names using NER"""
        if pd.isna(names_str) or not names_str:
            return []
        
        # Split by commas first
        raw_names = [name.strip() for name in str(names_str).split(',') if name.strip()]
        
        # Use NER to identify and normalize person names
        normalized_names = []
        
        if self.models['ner']:
            try:
                for name in raw_names:
                    doc = self.models['ner'](name)
                    person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                    
                    if person_entities:
                        # Use NER-identified person name
                        normalized_names.extend(person_entities)
                    else:
                        # Fall back to original if NER doesn't identify as person
                        normalized_names.append(name)
            except Exception as e:
                logger.debug(f"Name normalization failed: {e}")
                normalized_names = raw_names
        else:
            normalized_names = raw_names
        
        # Add name variations for search
        variations = []
        for name in normalized_names[:8]:  # Limit to avoid bloat
            variations.append(name)
            # Add last name for queries like "Nolan movies"
            parts = name.split()
            if len(parts) >= 2:
                variations.append(parts[-1])
        
        return list(set(variations))
    
    def generate_decade_era_tags(self, release_year: Any) -> List[str]:
        """Generate temporal tags using simple logic (no model needed)"""
        try:
            year = int(release_year) if pd.notna(release_year) else None
            if not year:
                return []
            
            tags = []
            decade = (year // 10) * 10
            tags.append(f"{decade}s")
            
            # Era classification
            if year >= 2020:
                tags.append("recent")
            elif year >= 2010:
                tags.append("modern")
            elif year >= 2000:
                tags.append("contemporary")  
            else:
                tags.append("classic")
            
            return tags
            
        except (ValueError, TypeError):
            return []
    
    def enrich_single_row(self, row: pd.Series) -> str:
        """Generate comprehensive model-based tags for a single row"""
        all_tags = []
        
        # 1. Extract themes from description using zero-shot classification
        description = row.get('description', '')
        themes = self.extract_themes_with_zero_shot(description)
        all_tags.extend(themes)
        
        # 2. Extract keywords from description using KeyBERT
        keywords = self.extract_keywords_with_keybert(description)
        all_tags.extend(keywords)
        
        # 3. Extract and normalize actor names using NER
        cast_variations = self.normalize_names_with_ner(row.get('cast', ''))
        all_tags.extend(cast_variations[:10])  # Limit actor tags
        
        # 4. Extract and normalize director names using NER
        director_variations = self.normalize_names_with_ner(row.get('director', ''))
        all_tags.extend(director_variations)
        
        # 5. Extract entities from description
        entities = self.extract_entities_with_ner(description)
        for entity_list in entities.values():
            all_tags.extend(entity_list[:3])  # Limit each entity type
        
        # 6. Detect mood
        mood = self.detect_mood_with_sentiment(description)
        if mood != "neutral":
            all_tags.append(mood)
        
        # 7. Add temporal tags
        temporal_tags = self.generate_decade_era_tags(row.get('release_year'))
        all_tags.extend(temporal_tags)
        
        # 8. Add original genres (already curated data)
        if pd.notna(row.get('listed_in')) and row.get('listed_in'):
            genres = [g.strip().lower() for g in str(row.get('listed_in')).split(',')]
            all_tags.extend(genres[:5])
        
        # Clean and deduplicate
        clean_tags = []
        for tag in all_tags:
            if tag and isinstance(tag, str) and len(tag.strip()) > 1:
                clean_tag = tag.strip().lower().replace(' ', '-')
                clean_tags.append(clean_tag)
        
        # Remove duplicates
        unique_tags = list(dict.fromkeys(clean_tags))  # Preserves order
        
        return ' | '.join(unique_tags[:25])  # Limit to 25 tags
    
    def enrich_dataset(self, input_csv: str, output_csv: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Enrich dataset with model-based tags"""
        
        logger.info(f"Loading dataset from {input_csv}")
        df = pd.read_csv(input_csv, encoding='latin-1').fillna('')
        
        # Use sample for testing if specified
        if sample_size:
            df = df.head(sample_size)
            logger.info(f"Using sample of {sample_size} rows for testing")
        
        logger.info(f"Processing {len(df)} rows with model-based tagging...")
        
        # Process each row
        semantic_tags = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} rows...")
            
            try:
                tags = self.enrich_single_row(row)
                semantic_tags.append(tags)
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                semantic_tags.append('')
        
        # Add semantic tags column
        df['semantic_tags'] = semantic_tags
        
        # Save enriched dataset
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"✅ Model-enriched dataset saved to {output_csv}")
        
        # Statistics
        non_empty = df[df['semantic_tags'] != '']
        avg_tags = non_empty['semantic_tags'].str.split('|').str.len().mean()
        
        logger.info("Model-Based Enrichment Statistics:")
        logger.info(f"- Rows processed: {len(df)}")
        logger.info(f"- Rows with tags: {len(non_empty)}")
        logger.info(f"- Average tags per row: {avg_tags:.1f}")
        
        return df


def main():
    """Main function - test with small sample first"""
    tagger = ModelBasedTagger()
    
    input_file = "data/netflix_titles_cleaned.csv"
    
    # First test with small sample
    sample_output = "data/netflix_sample_enriched.csv"
    logger.info("Testing with 50-row sample...")
    
    sample_df = tagger.enrich_dataset(input_file, sample_output, sample_size=50)
    
    # Show sample results
    print("\n🎯 Sample model-based enriched rows:")
    for i in range(3):
        if i < len(sample_df):
            row = sample_df.iloc[i]
            print(f"\nTitle: {row['title']}")
            print(f"Model Tags: {row['semantic_tags']}")
    
    # Ask user if they want to process full dataset
    print(f"\n✅ Sample processing complete. Check {sample_output} for results.")
    print("To process the full dataset, modify the main() function.")


if __name__ == "__main__":
    main()