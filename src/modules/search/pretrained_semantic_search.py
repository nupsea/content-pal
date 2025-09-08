"""
Pre-trained Semantic Search System using Cross-Encoder for Query-Document Understanding

Uses existing pre-trained models to understand semantic relationships without custom training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import logging

from .core import ContentSearchSystem, SearchResult, SearchConfig


class PretrainedSemanticSearch:
    """
    Search system that uses pre-trained semantic models for understanding
    query-document relationships without requiring custom training.
    """
    
    def __init__(self, backend_type: str = "minsearch", 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.backend_type = backend_type
        self.model_name = model_name
        
        # Base retrieval system for candidate generation
        self.base_search = ContentSearchSystem(backend_type=backend_type)
        
        # Pre-trained semantic model
        self.cross_encoder = None
        self.indexed = False
        
        # Document store for semantic analysis
        self.documents = {}
        
        # Initialize cross-encoder
        try:
            print(f"** Loading pre-trained semantic model: {model_name}")
            self.cross_encoder = CrossEncoder(model_name)
            print("** Semantic model loaded successfully")
        except Exception as e:
            print(f" !! Failed to load semantic model: {e}")
            self.cross_encoder = None
        
    def index_data(self, csv_path: str):
        """Index data and prepare document store"""
        self.base_search.index_data(csv_path)
        
        # Load and store document content for semantic analysis
        df = pd.read_csv(csv_path, encoding='latin-1').fillna('')
        
        for _, row in df.iterrows():
            doc_id = str(row['show_id'])
            
            # Create rich document representation for semantic matching
            doc_text = self._create_document_representation(row)
            
            self.documents[doc_id] = {
                'text': doc_text,
                'title': row['title'],
                'metadata': row.to_dict()
            }
        
        self.indexed = True
        print(f"Indexed {len(self.documents)} documents for semantic search")
    
    def _create_document_representation(self, row: pd.Series) -> str:
        """Create rich text representation of document for semantic understanding"""
        # Combine multiple fields into a coherent text representation
        parts = []
        
        # Title (most important)
        if pd.notna(row['title']) and row['title']:
            parts.append(f"Title: {row['title']}")
        
        # Description (crucial for semantic understanding)
        if pd.notna(row['description']) and row['description']:
            # Truncate long descriptions to fit model limits
            desc = str(row['description'])[:300]
            parts.append(f"Plot: {desc}")
        
        # Genres (important for categorization)
        if pd.notna(row['listed_in']) and row['listed_in']:
            parts.append(f"Genres: {row['listed_in']}")
        
        # Cast (important for actor-based queries)
        if pd.notna(row['cast']) and row['cast']:
            # Truncate long cast lists
            cast = str(row['cast'])[:150]
            parts.append(f"Cast: {cast}")
        
        # Additional context
        if pd.notna(row['type']) and row['type']:
            parts.append(f"Type: {row['type']}")
        
        if pd.notna(row['release_year']):
            parts.append(f"Year: {row['release_year']}")
        
        return " | ".join(parts)
    
    def semantic_rerank(self, query: str, candidates: List[SearchResult], top_k: int = 50) -> List[SearchResult]:
        """
        Use pre-trained semantic model to rerank candidates based on semantic similarity
        """
        if not self.cross_encoder:
            # Fall back to original ranking if no semantic model
            return candidates[:top_k]
        
        # Prepare query-document pairs for scoring
        pairs = []
        valid_candidates = []
        
        for candidate in candidates:
            if candidate.id in self.documents:
                doc_text = self.documents[candidate.id]['text']
                # Truncate to fit model context window
                if len(doc_text) > 400:
                    doc_text = doc_text[:400] + "..."
                
                pairs.append([query, doc_text])
                valid_candidates.append(candidate)
        
        if not pairs:
            return candidates[:top_k]
        
        # Get semantic similarity scores
        try:
            scores = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            
            # Update candidate scores with semantic scores
            for i, (candidate, semantic_score) in enumerate(zip(valid_candidates, scores)):
                # Combine original score with semantic score
                # Give semantic score higher weight since it understands meaning better
                original_score = candidate.score if hasattr(candidate, 'score') else 1.0
                
                # Normalize scores to 0-1 range and combine
                normalized_original = min(original_score / 10.0, 1.0)  # Assuming original scores are roughly 0-10
                normalized_semantic = max(0.0, min(float(semantic_score), 1.0))  # Clamp to 0-1
                
                # Weight combination: 30% original, 70% semantic
                candidate.score = 0.3 * normalized_original + 0.7 * normalized_semantic
            
            # Sort by combined score
            valid_candidates.sort(key=lambda x: x.score, reverse=True)
            
            return valid_candidates[:top_k]
            
        except Exception as e:
            print(f"!! Semantic reranking failed: {e}")
            return candidates[:top_k]
    
    def search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """
        Main search method combining base retrieval with pre-trained semantic understanding
        """
        if not self.indexed:
            raise RuntimeError("Data not indexed. Call index_data() first.")
        
        # Step 1: Get initial candidates using base search with high recall
        # Use higher recall to get more candidates for semantic reranking
        candidate_multiplier = 3  # Get 3x more candidates than needed
        base_candidates = min(200, top_k * candidate_multiplier)
        
        base_config = SearchConfig(
            boost_weights={
                'title': 3.0,
                'cast': 2.5, 
                'director': 2.0,
                'listed_in': 2.0, 
                'description': 4.0  # Boost description for semantic content
            },
            max_results=base_candidates
        )
        
        base_results = self.base_search.search(query, base_config)
        
        # Step 2: Apply semantic reranking
        if self.cross_encoder and base_results:
            final_results = self.semantic_rerank(query, base_results, top_k)
        else:
            final_results = base_results[:top_k]
        
        return final_results