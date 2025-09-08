"""
Enriched Semantic Search System

Uses model-enriched data with semantic tags for superior search performance
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import logging

from .core import ContentSearchSystem, SearchResult, SearchConfig

logger = logging.getLogger(__name__)


class EnrichedSemanticSearch:
    """
    Semantic search system that leverages model-generated semantic tags
    """
    
    def __init__(self, backend_type: str = "minsearch", 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_enriched_data: bool = True):
        self.backend_type = backend_type
        self.model_name = model_name
        self.use_enriched_data = use_enriched_data
        
        # Base retrieval system
        self.base_search = ContentSearchSystem(backend_type=backend_type)
        
        # Semantic model
        self.cross_encoder = None
        self.indexed = False
        
        # Document store with enriched data
        self.documents = {}
        
        # Initialize cross-encoder
        try:
            print(f"** Loading enriched semantic model: {model_name}")
            self.cross_encoder = CrossEncoder(model_name)
            print("** Enriched semantic model loaded successfully")
        except Exception as e:
            print(f"!! Failed to load semantic model: {e}")
            self.cross_encoder = None
    
    def index_data(self, csv_path: str):
        """Index enriched data with semantic tags"""
        self.base_search.index_data(csv_path)
        
        # Load enriched data
        df = pd.read_csv(csv_path, encoding='utf-8').fillna('')  # Use utf-8 for enriched data
        
        for _, row in df.iterrows():
            doc_id = str(row['show_id'])
            
            # Create enhanced document representation using semantic tags
            doc_text = self._create_enriched_document_representation(row)
            
            self.documents[doc_id] = {
                'text': doc_text,
                'title': row['title'],
                'semantic_tags': row.get('semantic_tags', ''),
                'metadata': row.to_dict()
            }
        
        self.indexed = True
        print(f"** Indexed {len(self.documents)} documents with semantic enrichment")
        
        # Show enrichment stats
        enriched_docs = sum(1 for doc in self.documents.values() if doc['semantic_tags'])
        avg_tags = np.mean([len(doc['semantic_tags'].split('|')) for doc in self.documents.values() if doc['semantic_tags']])
        print(f"** Enriched documents: {enriched_docs}/{len(self.documents)}")
        print(f"** Average tags per document: {avg_tags:.1f}")
    
    def _create_enriched_document_representation(self, row: pd.Series) -> str:
        """Create document representation leveraging model-generated semantic tags"""
        parts = []
        
        # Title (highest priority)
        if pd.notna(row['title']) and row['title']:
            parts.append(f"Title: {row['title']}")
        
        # Director
        if pd.notna(row['director']) and row['director']:
            parts.append(f"Director: {row['director']}")
        
        # Cast (limited to avoid bloat)
        if pd.notna(row['cast']) and row['cast']:
            cast = str(row['cast'])[:100]  # Truncate long cast lists
            parts.append(f"Cast: {cast}")
        
        # Original genres
        if pd.notna(row['listed_in']) and row['listed_in']:
            parts.append(f"Genres: {row['listed_in']}")
        
        # Description
        if pd.notna(row['description']) and row['description']:
            desc = str(row['description'])[:250]
            parts.append(f"Plot: {desc}")
        
        # SEMANTIC TAGS - The key enhancement!
        if self.use_enriched_data and pd.notna(row.get('semantic_tags')) and row.get('semantic_tags'):
            # Clean and format semantic tags for better matching
            semantic_tags = str(row['semantic_tags']).replace('|', ' ')
            parts.append(f"Tags: {semantic_tags}")
        
        # Year for temporal queries
        if pd.notna(row['release_year']):
            parts.append(f"Year: {row['release_year']}")
        
        # Type
        if pd.notna(row['type']) and row['type']:
            parts.append(f"Type: {row['type']}")
        
        return " | ".join(parts)
    
    def search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """Enhanced search leveraging semantic tags"""
        if not self.indexed:
            raise RuntimeError("Data not indexed. Call index_data() first.")
        
        # Removed verbose logging for cleaner evaluation output
        
        # Enhanced search configuration that considers semantic tags
        base_config = SearchConfig(
            boost_weights={
                'title': 4.0,       # High title relevance
                'director': 4.0,    # High director relevance for "Nolan movies" queries
                'cast': 2.5,        # Actor relevance
                'listed_in': 3.0,   # Genre relevance
                'description': 3.5, # Content relevance
                'semantic_tags': 5.0  # HIGHEST boost for model-generated tags!
            },
            max_results=min(200, top_k * 4)
        )
        
        # Get base results
        try:
            base_results = self.base_search.search(query, base_config)
        except Exception as e:
            print(f"!! Base search failed, trying fallback: {e}")
            # Fallback without semantic_tags field if it doesn't exist in index
            fallback_config = SearchConfig(
                boost_weights={
                    'title': 4.0,
                    'director': 4.0,
                    'cast': 2.5,
                    'listed_in': 3.0,
                    'description': 3.5
                },
                max_results=min(200, top_k * 4)
            )
            base_results = self.base_search.search(query, fallback_config)
        
        # Apply semantic reranking with enriched document representations
        if self.cross_encoder and base_results:
            final_results = self.semantic_rerank(query, base_results, top_k)
        else:
            final_results = base_results[:top_k]
        
        return final_results
    
    def semantic_rerank(self, query: str, candidates: List[SearchResult], top_k: int = 50) -> List[SearchResult]:
        """Enhanced semantic reranking using enriched document representations"""
        if not self.cross_encoder:
            return candidates[:top_k]
        
        pairs = []
        valid_candidates = []
        
        for candidate in candidates:
            if candidate.id in self.documents:
                # Use enriched document representation (includes semantic tags)
                doc_text = self.documents[candidate.id]['text']
                
                # Truncate while preserving key information
                if len(doc_text) > 450:
                    # Preserve title, director, and tags if possible
                    parts = doc_text.split(' | ')
                    key_parts = []
                    other_parts = []
                    
                    for part in parts:
                        if any(keyword in part.lower() for keyword in ['title:', 'director:', 'tags:']):
                            key_parts.append(part)
                        else:
                            other_parts.append(part)
                    
                    # Combine key parts with truncated others
                    key_text = ' | '.join(key_parts)
                    remaining_space = 450 - len(key_text)
                    
                    if remaining_space > 0 and other_parts:
                        other_text = ' | '.join(other_parts)[:remaining_space]
                        doc_text = f"{key_text} | {other_text}"
                    else:
                        doc_text = key_text
                
                pairs.append([query, doc_text])
                valid_candidates.append(candidate)
        
        if not pairs:
            return candidates[:top_k]
        
        try:
            scores = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            
            # Enhanced scoring with semantic tag boost
            for i, (candidate, semantic_score) in enumerate(zip(valid_candidates, scores)):
                original_score = candidate.score if hasattr(candidate, 'score') else 1.0
                
                # Check if document has rich semantic tags for additional boost
                doc = self.documents[candidate.id]
                semantic_tag_boost = 0.0
                if doc['semantic_tags'] and len(doc['semantic_tags'].split('|')) > 10:
                    semantic_tag_boost = 0.05  # Small boost for well-tagged documents
                
                # Normalize and combine scores
                normalized_original = min(original_score / 10.0, 1.0)
                normalized_semantic = max(0.0, min(float(semantic_score), 1.0))
                
                # 25% original + 75% semantic + tag boost
                candidate.score = (0.25 * normalized_original + 
                                 0.75 * normalized_semantic + 
                                 semantic_tag_boost)
            
            # Sort by enhanced score
            valid_candidates.sort(key=lambda x: x.score, reverse=True)
            return valid_candidates[:top_k]
            
        except Exception as e:
            print(f"!! Enriched semantic reranking failed: {e}")
            return candidates[:top_k]


def create_enriched_search_system(enriched_csv_path: str = "data/netflix_sample_enriched.csv") -> EnrichedSemanticSearch:
    """Factory function to create enriched search system"""
    system = EnrichedSemanticSearch(backend_type="minsearch")
    
    try:
        system.index_data(enriched_csv_path)
        return system
    except Exception as e:
        print(f"!! Failed to load enriched data from {enriched_csv_path}: {e}")
        print("!! Falling back to original data without enrichment")
        
        # Fallback to non-enriched system
        system.use_enriched_data = False
        system.index_data("data/netflix_titles_cleaned.csv")
        return system