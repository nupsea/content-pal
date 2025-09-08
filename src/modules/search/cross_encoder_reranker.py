"""
Cross-Encoder Reranking Module

Provides semantic reranking using cross-encoder models that can be integrated 
into any search system. Based on the successful PretrainedSemantic_CrossEncoder approach.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import logging

from .core import SearchResult


class CrossEncoderReranker:
    """
    Cross-encoder based semantic reranker that can enhance any search system.
    Provides the semantic understanding that made PretrainedSemantic_CrossEncoder successful.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.cross_encoder = None
        self.documents = {}
        self.ready = False
        
        # Initialize cross-encoder model
        try:
            print(f"** Loading cross-encoder reranker: {model_name}")
            self.cross_encoder = CrossEncoder(model_name)
            print("** Cross-encoder reranker loaded successfully")
            self.ready = True
        except Exception as e:
            print(f"!! Failed to load cross-encoder reranker: {e}")
            self.cross_encoder = None
            self.ready = False
    
    def prepare_documents(self, csv_path: str):
        """Prepare document representations for semantic reranking"""
        if not self.ready:
            return
            
        df = pd.read_csv(csv_path, encoding='latin-1').fillna('')
        
        for _, row in df.iterrows():
            doc_id = str(row['show_id'])
            doc_text = self._create_document_text(row)
            
            self.documents[doc_id] = {
                'text': doc_text,
                'title': row['title'],
                'metadata': row.to_dict()
            }
        
        print(f"** Cross-encoder prepared {len(self.documents)} documents")
    
    def _create_document_text(self, row: pd.Series) -> str:
        """
        Create optimal document representation for cross-encoder reranking.
        Based on the successful PretrainedSemantic approach.
        """
        parts = []
        
        # Title (highest priority for matching)
        if pd.notna(row['title']) and row['title']:
            parts.append(f"Title: {row['title']}")
        
        # Description (essential for semantic content understanding)
        if pd.notna(row['description']) and row['description']:
            desc = str(row['description'])[:300]  # Truncate for model limits
            parts.append(f"Plot: {desc}")
        
        # Genres (important for categorization)
        if pd.notna(row['listed_in']) and row['listed_in']:
            parts.append(f"Genres: {row['listed_in']}")
        
        # Cast (crucial for actor-based searches)
        if pd.notna(row['cast']) and row['cast']:
            cast = str(row['cast'])[:150]  # Truncate long cast lists
            parts.append(f"Cast: {cast}")
        
        # Director (important for director queries)
        if pd.notna(row['director']) and row['director']:
            director = str(row['director'])[:100]
            parts.append(f"Director: {director}")
        
        # Metadata
        if pd.notna(row['type']) and row['type']:
            parts.append(f"Type: {row['type']}")
            
        if pd.notna(row['release_year']):
            parts.append(f"Year: {row['release_year']}")
        
        # Enhanced semantic tags (if available from enriched datasets)
        if 'semantic_tags' in row and pd.notna(row['semantic_tags']) and row['semantic_tags']:
            semantic_tags = str(row['semantic_tags']).replace('|', ',')[:150]
            parts.append(f"Tags: {semantic_tags}")
        
        # Enhanced tags from no-API enrichment
        if 'no_api_enhanced_tags' in row and pd.notna(row['no_api_enhanced_tags']) and row['no_api_enhanced_tags']:
            enhanced_tags = str(row['no_api_enhanced_tags']).replace('|', ',')[:150]
            parts.append(f"Enhanced: {enhanced_tags}")
        
        return " | ".join(parts)
    
    def rerank(self, query: str, candidates: List[SearchResult], 
               top_k: int = 50, semantic_weight: float = 0.7) -> List[SearchResult]:
        """
        Apply cross-encoder semantic reranking to search results.
        
        This is the core method that provides the semantic understanding boost
        that made PretrainedSemantic_CrossEncoder the best performer.
        
        Args:
            query: Search query
            candidates: Initial search results to rerank
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (0.7 = 70% semantic, 30% original)
        """
        if not self.ready or not self.cross_encoder or not candidates:
            return candidates[:top_k]
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        valid_candidates = []
        
        for candidate in candidates:
            if candidate.id in self.documents:
                doc_text = self.documents[candidate.id]['text']
                
                # Truncate to fit model context window (cross-encoders have limits)
                if len(doc_text) > 400:
                    doc_text = doc_text[:400] + "..."
                
                pairs.append([query, doc_text])
                valid_candidates.append(candidate)
        
        if not pairs:
            return candidates[:int(top_k)]
        
        try:
            # Get semantic similarity scores from cross-encoder
            scores = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            
            # Combine original and semantic scores (proven optimal combination)
            for candidate, semantic_score in zip(valid_candidates, scores):
                original_score = getattr(candidate, 'score', 1.0)
                
                # Normalize scores to 0-1 range
                normalized_original = min(abs(original_score) / 10.0, 1.0)
                normalized_semantic = max(0.0, min(float(semantic_score), 1.0))
                
                # Apply proven weighting: 30% original + 70% semantic
                candidate.score = (1 - semantic_weight) * normalized_original + semantic_weight * normalized_semantic
            
            # Sort by enhanced scores
            valid_candidates.sort(key=lambda x: x.score, reverse=True)
            return valid_candidates[:int(top_k)]
            
        except Exception as e:
            print(f"!! Cross-encoder reranking failed: {e}")
            return candidates[:int(top_k)]


# Global reranker instance (singleton pattern for efficiency)
_global_reranker = None

def get_cross_encoder_reranker() -> CrossEncoderReranker:
    """Get global cross-encoder reranker instance"""
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = CrossEncoderReranker()
    return _global_reranker

def prepare_cross_encoder_reranking(csv_path: str):
    """Prepare cross-encoder reranking for dataset"""
    reranker = get_cross_encoder_reranker()
    reranker.prepare_documents(csv_path)

def apply_cross_encoder_reranking(query: str, results: List[SearchResult], 
                                 top_k: int = 50, semantic_weight: float = 0.7) -> List[SearchResult]:
    """Apply cross-encoder semantic reranking to results"""
    reranker = get_cross_encoder_reranker()
    return reranker.rerank(query, results, top_k, semantic_weight)