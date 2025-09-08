"""
UltraBoost Search System

Ensemble system combining best search strategies for maximum performance.
Extracted from evaluation script for cleaner architecture.
"""

from typing import List, Dict, Any, Optional
from .strategic_search import StrategicSearchSystem, StrategicSearchConfig
from .pretrained_semantic_search import PretrainedSemanticSearch
from .enriched_semantic_search import EnrichedSemanticSearch
from .cross_encoder_reranker import get_cross_encoder_reranker
from .core import SearchResult


class UltraBoostSearchSystem:
    """
    Ultra-boosted ensemble search system combining multiple strategies
    
    Performance: 34.6% HR@10
    Strategy: Weighted ensemble of best components + cross-encoder reranking
    """
    
    def __init__(self, backend_type: str = "minsearch"):
        self.backend_type = backend_type
        self.indexed = False
        
        # Initialize component systems
        self.enriched_system = None
        self.strategic_system = None
        self.pretrained_system = None
    
    def index_data(self, csv_path: str):
        """Index data across all component systems"""
        # Use multiple search strategies
        self.enriched_system = EnrichedSemanticSearch(backend_type=self.backend_type)
        self.enriched_system.index_data(csv_path=csv_path)
        
        self.strategic_system = StrategicSearchSystem(backend_type=self.backend_type)
        self.strategic_system.index_data(csv_path=csv_path)
        
        self.pretrained_system = PretrainedSemanticSearch(backend_type=self.backend_type)
        self.pretrained_system.index_data(csv_path=csv_path)
        
        self.indexed = True
        print(f"** UltraBoost indexed data across 3 component systems")
    
    def search(self, query: str, top_k: int = 50, config=None) -> List[SearchResult]:
        """
        Main search method combining multiple strategies with weighted fusion
        
        Args:
            query: Search query
            top_k: Number of results to return
            config: Optional configuration (for compatibility)
            
        Returns:
            List of SearchResult objects ranked by combined score
        """
        if not self.indexed:
            raise RuntimeError("Data not indexed. Call index_data() first.")
        
        top_k_int = top_k
        
        # System 1: Enriched semantic (high recall with tags)
        try:
            enriched_results = self.enriched_system.search(query, top_k=min(200, top_k_int * 4))
        except Exception as e:
            print(f"UltraBoost enriched search failed: {e}")
            enriched_results = []
        
        # System 2: Strategic (query understanding)  
        try:
            strategic_config = StrategicSearchConfig(final_result_count=min(150, top_k_int * 3))
            strategic_results = self.strategic_system.search(query, strategic_config)
        except Exception as e:
            print(f"UltraBoost strategic search failed: {e}")
            strategic_results = []
        
        # System 3: Pretrained semantic (proven best performer)
        try:
            pretrained_results = self.pretrained_system.search(query, top_k=min(100, top_k_int * 2))
        except Exception as e:
            print(f"UltraBoost pretrained search failed: {e}")
            pretrained_results = []
        
        # Combine with sophisticated weighted fusion
        combined_results = {}
        
        # Add pretrained results (highest weight - proven best)
        for i, result in enumerate(pretrained_results):
            score = (len(pretrained_results) - i) / len(pretrained_results) * 0.5  # 50% weight
            combined_results[result.id] = result
            result.score = score
        
        # Add enriched results (high recall)
        for i, result in enumerate(enriched_results):
            base_score = (len(enriched_results) - i) / len(enriched_results) * 0.3  # 30% weight
            if result.id in combined_results:
                combined_results[result.id].score += base_score
            else:
                result.score = base_score
                combined_results[result.id] = result
        
        # Add strategic results (query understanding)
        for i, result in enumerate(strategic_results):
            base_score = (len(strategic_results) - i) / len(strategic_results) * 0.2  # 20% weight
            if result.id in combined_results:
                combined_results[result.id].score += base_score
            else:
                result.score = base_score
                combined_results[result.id] = result
        
        # Sort by combined score
        final_candidates = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        
        # Apply cross-encoder reranking for final boost
        if final_candidates:
            try:
                reranker = get_cross_encoder_reranker()
                final_results = reranker.rerank(
                    query, 
                    final_candidates[:min(200, len(final_candidates))],  # Rerank top 200
                    top_k=top_k_int,
                    semantic_weight=0.85  # 85% semantic for maximum performance
                )
            except Exception as e:
                print(f"UltraBoost reranking failed: {e}")
                final_results = final_candidates[:top_k_int]
        else:
            final_results = []
        
        return final_results


def create_ultraboost_search_system(csv_path: str = "data/netflix_titles_enriched_full.csv") -> UltraBoostSearchSystem:
    """Factory function to create UltraBoost search system"""
    system = UltraBoostSearchSystem(backend_type="minsearch")
    system.index_data(csv_path)
    return system