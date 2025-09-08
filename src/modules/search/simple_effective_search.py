"""
Simple Effective Search System

Based on UltraBoost analysis: Focus on what works instead of over-engineering.
Key insight: PretrainedSemanticSearch (50% weight in UltraBoost) is the proven performer.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import logging

from .core import ContentSearchSystem, SearchResult, SearchConfig

logger = logging.getLogger(__name__)


class SimpleEffectiveSearch:
    """
    Simple but effective search system targeting >50% HR@10
    
    Based on analysis of UltraBoost (34.6% HR@10) - focuses on proven components:
    1. Solid base retrieval with balanced field weights  
    2. Cross-encoder semantic reranking (proven effective)
    3. High recall candidate generation
    4. No over-engineering or complex patterns
    """
    
    def __init__(self, backend_type: str = "minsearch", 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.backend_type = backend_type
        self.model_name = model_name
        
        # Simple, proven base retrieval
        self.base_search = ContentSearchSystem(backend_type=backend_type)
        
        # Proven semantic model (same as PretrainedSemanticSearch)
        self.cross_encoder = None
        self.indexed = False
        
        # Document store for semantic reranking
        self.documents = {}
        
        # Load proven semantic model
        try:
            self.cross_encoder = CrossEncoder(model_name)
        except Exception as e:
            print(f"!! Failed to load semantic model: {e}")
            self.cross_encoder = None
    
    def index_data(self, csv_path: str):
        """Index data with focus on clean, effective retrieval"""
        self.base_search.index_data(csv_path)
        
        # Load documents for semantic reranking (same as PretrainedSemanticSearch)
        df = pd.read_csv(csv_path, encoding='latin-1').fillna('')
        
        for _, row in df.iterrows():
            doc_id = str(row['show_id'])
            
            # Create clean document representation focused on key fields
            doc_text = self._create_document_representation(row)
            
            self.documents[doc_id] = {
                'text': doc_text,
                'title': row['title'],
                'metadata': row.to_dict()
            }
        
        self.indexed = True
        print(f"** Indexed {len(self.documents)} documents for simple effective search")
    
    def _create_document_representation(self, row: pd.Series) -> str:
        """Create clean document representation focused on core content"""
        parts = []
        
        # Title (highest priority - exact match queries)
        if pd.notna(row['title']) and row['title']:
            parts.append(f"Title: {row['title']}")
        
        # Description (crucial for semantic matching)  
        if pd.notna(row['description']) and row['description']:
            desc = str(row['description'])[:300]  # Truncate but keep full context
            parts.append(f"Plot: {desc}")
        
        # Genres (important for category queries)
        if pd.notna(row['listed_in']) and row['listed_in']:
            parts.append(f"Genres: {row['listed_in']}")
        
        # Cast (important for actor queries)
        if pd.notna(row['cast']) and row['cast']:
            cast = str(row['cast'])[:150]  # Top cast members
            parts.append(f"Cast: {cast}")
        
        # Director (important for director-based queries)
        if pd.notna(row['director']) and row['director']:
            parts.append(f"Director: {row['director']}")
        
        # Type and Year for basic filtering
        if pd.notna(row['type']) and row['type']:
            parts.append(f"Type: {row['type']}")
        
        if pd.notna(row['release_year']):
            parts.append(f"Year: {row['release_year']}")
        
        return " | ".join(parts)
    
    def search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """
        Simple effective search targeting >50% HR@10
        
        Strategy:
        1. High recall base retrieval with balanced weights
        2. Semantic reranking with proven cross-encoder
        3. Focus on fundamentals, avoid over-engineering
        """
        if not self.indexed:
            raise RuntimeError("Data not indexed. Call index_data() first.")
        
        # Step 1: High recall base retrieval with balanced, proven weights
        # Based on PretrainedSemanticSearch config (from UltraBoost)
        recall_multiplier = 4  # Get 4x candidates for reranking
        base_candidates = min(200, top_k * recall_multiplier)
        
        # Proven field weights from PretrainedSemanticSearch  
        base_config = SearchConfig(
            boost_weights={
                'title': 3.0,           # Good title boost (not over-aggressive)
                'description': 4.0,     # Boost description for semantic content
                'cast': 2.5,           # Actor queries
                'director': 2.0,       # Director queries
                'listed_in': 2.0,     # Genre queries
            },
            max_results=base_candidates
        )
        
        # Get base results with high recall
        base_results = self.base_search.search(query, base_config)
        
        # Step 2: Apply proven semantic reranking (same as PretrainedSemanticSearch)
        if self.cross_encoder and base_results:
            final_results = self._semantic_rerank(query, base_results, top_k)
        else:
            final_results = base_results[:top_k]
        
        return final_results
    
    def _semantic_rerank(self, query: str, candidates: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Apply proven semantic reranking (same approach as PretrainedSemanticSearch)"""
        if not self.cross_encoder:
            return candidates[:top_k]
        
        # Prepare query-document pairs
        pairs = []
        valid_candidates = []
        
        for candidate in candidates:
            if candidate.id in self.documents:
                doc_text = self.documents[candidate.id]['text']
                
                # Smart truncation preserving key information
                if len(doc_text) > 400:
                    doc_text = doc_text[:400] + "..."
                
                pairs.append([query, doc_text])
                valid_candidates.append(candidate)
        
        if not pairs:
            return candidates[:top_k]
        
        try:
            # Get semantic scores
            scores = self.cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            
            # Apply proven score combination (30% original + 70% semantic)
            for i, (candidate, semantic_score) in enumerate(zip(valid_candidates, scores)):
                original_score = candidate.score if hasattr(candidate, 'score') else 1.0
                
                # Normalize scores
                normalized_original = min(original_score / 10.0, 1.0)
                normalized_semantic = max(0.0, min(float(semantic_score), 1.0))
                
                # Proven combination weights from PretrainedSemanticSearch
                candidate.score = 0.3 * normalized_original + 0.7 * normalized_semantic
            
            # Sort by combined score
            valid_candidates.sort(key=lambda x: x.score, reverse=True)
            
            return valid_candidates[:top_k]
            
        except Exception as e:
            print(f"!! Semantic reranking failed: {e}")
            return candidates[:top_k]


def create_simple_effective_search_system(csv_path: str = "data/netflix_titles_enriched_full.csv") -> SimpleEffectiveSearch:
    """Factory function to create simple effective search system"""
    system = SimpleEffectiveSearch(backend_type="minsearch")
    
    try:
        system.index_data(csv_path)
        return system
    except Exception as e:
        print(f"!! Failed to create simple effective search: {e}")
        raise