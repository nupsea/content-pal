#!/usr/bin/env python3
"""
Adaptive retrieval system that selects search strategy based on query intent.
Optimizes for different query types to improve MRR and Hit Rate.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from opensearchpy import OpenSearch
from search.index_assets import make_client
from search.hybrid_search import hybrid_search, bm25_prf, knn_candidates, rrf_fuse, rerank_topk
from rag.query_classifier import QueryClassifier, QueryIntent


class AdaptiveRetriever:
    """
    Smart retrieval system that adapts strategy based on query intent.
    
    Different strategies:
    - SPECIFIC queries: Keyword-heavy search with exact matching
    - GENRE queries: Filtered search with category matching  
    - MOOD queries: Semantic search with description emphasis
    - TEMPORAL queries: Filtered search with date constraints
    - COMPLEX queries: Full semantic search with cross-encoder reranking
    - HYBRID queries: Balanced hybrid approach with dynamic weights
    """
    
    def __init__(self, client: Optional[OpenSearch] = None, index: str = "netflix_assets_v6"):
        self.client = client or make_client()
        self.index = index
        self.classifier = QueryClassifier()
    
    def retrieve(self, query: str, top_k: int = 50, use_reranking: bool = False) -> Dict:
        """
        Main retrieval method with adaptive strategy selection.
        
        Args:
            query: User query string
            top_k: Number of results to return
            use_reranking: Whether to apply cross-encoder reranking
            
        Returns:
            Dict with hits and metadata about the retrieval strategy used
        """
        # Classify the query
        intent = self.classifier.classify(query)
        
        # Select and execute retrieval strategy
        if intent.intent_type == "SPECIFIC":
            results = self._retrieve_specific(query, intent, top_k)
        elif intent.intent_type == "GENRE":
            results = self._retrieve_genre(query, intent, top_k)
        elif intent.intent_type == "MOOD":
            results = self._retrieve_mood(query, intent, top_k)
        elif intent.intent_type == "TEMPORAL":
            results = self._retrieve_temporal(query, intent, top_k)
        elif intent.intent_type == "COMPLEX":
            results = self._retrieve_complex(query, intent, top_k)
        else:  # HYBRID or unknown
            results = self._retrieve_hybrid(query, intent, top_k)
        
        # Apply cross-encoder reranking if requested
        if use_reranking and results.get("hits"):
            reranked_hits = rerank_topk(query, results["hits"], k=min(top_k, 20))
            results["hits"] = reranked_hits
            results["reranked"] = True
        
        # Add metadata
        results["query_intent"] = intent
        results["strategy_used"] = intent.search_strategy
        
        return results
    
    def _retrieve_specific(self, query: str, intent: QueryIntent, top_k: int) -> Dict:
        """Strategy for SPECIFIC queries (actor/director names)"""
        
        # Build targeted query focusing on cast/director fields
        body = {
            "size": top_k,
            "query": {
                "dis_max": {
                    "tie_breaker": 0.0,  # Winner takes all for specific matches
                    "queries": [
                        # Exact actor/director match (highest priority)
                        {
                            "multi_match": {
                                "query": query,
                                "type": "phrase",
                                "fields": ["cast^10", "director^10"],
                                "boost": 5.0
                            }
                        },
                        # Fuzzy actor/director match
                        {
                            "multi_match": {
                                "query": query,
                                "type": "best_fields",
                                "fields": ["cast^8", "director^8"],
                                "fuzziness": "AUTO",
                                "boost": 3.0
                            }
                        },
                        # Title match (in case they mention the title)
                        {
                            "multi_match": {
                                "query": query,
                                "type": "phrase_prefix",
                                "fields": ["title^6"],
                                "boost": 2.0
                            }
                        }
                    ]
                }
            }
        }
        
        response = self.client.search(index=self.index, body=body)
        return {"hits": response.get("hits", {}).get("hits", [])}
    
    def _retrieve_genre(self, query: str, intent: QueryIntent, top_k: int) -> Dict:
        """Strategy for GENRE queries - use hybrid approach instead of restrictive filtering"""
        
        # GENRE queries often fail with strict filtering, so use enhanced hybrid search
        # with boosted genre fields but no hard filters
        
        return hybrid_search(
            self.client, self.index, query, 
            top_k=top_k, 
            bm25_seed=100,    # More initial candidates
            bm25_final=400,   # Larger BM25 pool
            ann_k=300         # Balanced ANN
        )
    
    def _retrieve_mood(self, query: str, intent: QueryIntent, top_k: int) -> Dict:
        """Strategy for MOOD queries - use hybrid with semantic bias"""
        
        # MOOD queries benefit from semantic understanding but still need keyword matching
        return hybrid_search(
            self.client, self.index, query, 
            top_k=top_k, 
            bm25_seed=80,    # Standard seed
            bm25_final=300,  # Smaller BM25 pool
            ann_k=500        # Larger semantic pool
        )
    
    def _retrieve_temporal(self, query: str, intent: QueryIntent, top_k: int) -> Dict:
        """Strategy for TEMPORAL queries - hybrid with recency boost"""
        
        # TEMPORAL queries are tricky - often fail with strict filters
        # Use hybrid search with recency preferences
        return hybrid_search(
            self.client, self.index, query, 
            top_k=top_k, 
            bm25_seed=120,   # Larger seed for temporal diversity
            bm25_final=400,  # Large BM25 pool
            ann_k=300        # Standard ANN
        )
    
    def _retrieve_complex(self, query: str, intent: QueryIntent, top_k: int) -> Dict:
        """Strategy for COMPLEX queries - full semantic search"""
        
        # Pure semantic search works best for complex narrative queries
        semantic_results = knn_candidates(self.client, self.index, query, k=top_k*3)
        
        # Get top candidates for reranking
        candidates = semantic_results.get("hits", {}).get("hits", [])[:top_k*2]
        
        return {"hits": candidates}
    
    def _retrieve_hybrid(self, query: str, intent: QueryIntent, top_k: int) -> Dict:
        """Strategy for HYBRID and unknown queries - balanced approach"""
        
        # Use the existing hybrid search but with adjusted weights based on confidence
        confidence = intent.confidence
        
        # Lower confidence = rely more on keyword search
        # Higher confidence = allow more semantic search
        if confidence < 0.6:
            # Conservative approach - keyword heavy
            return hybrid_search(
                self.client, self.index, query, 
                top_k=top_k, bm25_seed=100, bm25_final=400, ann_k=300
            )
        else:
            # More balanced approach
            return hybrid_search(
                self.client, self.index, query, 
                top_k=top_k, bm25_seed=80, bm25_final=350, ann_k=400
            )


# Convenience function for backward compatibility
def adaptive_search(query: str, top_k: int = 50, use_reranking: bool = False, 
                   client: Optional[OpenSearch] = None, index: str = "netflix_assets_v6") -> Dict:
    """
    Perform adaptive retrieval on a single query.
    """
    retriever = AdaptiveRetriever(client=client, index=index)
    return retriever.retrieve(query, top_k=top_k, use_reranking=use_reranking)


if __name__ == "__main__":
    # Test the adaptive retriever
    test_queries = [
        "Bruce Greenwood movies",  # SPECIFIC
        "romantic comedies",       # GENRE  
        "feel-good family movies", # MOOD
        "movies added in 2023",    # TEMPORAL
        "story about a chess prodigy who overcomes challenges", # COMPLEX
        "thriller movies with suspense"  # HYBRID
    ]
    
    print("Testing Adaptive Retrieval System")
    print("=" * 50)
    
    retriever = AdaptiveRetriever()
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = retriever.retrieve(query, top_k=5)
        
        intent = result["query_intent"]
        print(f"Intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        print(f"Strategy: {result['strategy_used']}")
        print(f"Results: {len(result['hits'])} hits")
        
        # Show top result
        if result["hits"]:
            top_hit = result["hits"][0]
            source = top_hit.get("_source", {})
            print(f"Top result: {source.get('title', 'Unknown')} ({source.get('type', 'Unknown')})")
            print(f"Score: {top_hit.get('_score', 0):.3f}")
        
        print("-" * 30)