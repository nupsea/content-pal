#!/usr/bin/env python3
"""
Simple, robust search system focused on high Hit Rate and MRR.
No over-engineering - just effective keyword + semantic hybrid search.
"""

import os
import sys
from typing import Dict, List, Optional
from pathlib import Path
import re

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from opensearchpy import OpenSearch
from search.index_assets import make_client
from search.hybrid_search import _qvec, _EMB

class SimpleSearch:
    """
    Simple but effective search system.
    
    Core principles:
    1. Favor keyword matching for exact matches (actors, titles)
    2. Use semantic search for conceptual queries  
    3. Combine both with conservative weights
    4. No over-complex query analysis - let the search engines do the work
    """
    
    def __init__(self, client: Optional[OpenSearch] = None, index: str = "netflix_assets_v6"):
        self.client = client or make_client()
        self.index = index
        
    def search(self, query: str, top_k: int = 50) -> Dict:
        """
        Main search method - simple but robust hybrid approach.
        """
        
        # Clean query
        query = query.strip().lower()
        if not query:
            return {"hits": []}
            
        # Strategy: Multi-pronged search with different strengths
        
        # 1. EXACT/PHRASE matching (for titles, actors)
        phrase_results = self._phrase_search(query, top_k)
        
        # 2. KEYWORD matching (flexible term matching)  
        keyword_results = self._keyword_search(query, top_k)
        
        # 3. SEMANTIC matching (for conceptual queries)
        semantic_results = self._semantic_search(query, top_k)
        
        # 4. FUZZY matching (for typos, partial matches)
        fuzzy_results = self._fuzzy_search(query, top_k)
        
        # Combine results using weighted scoring
        combined_hits = self._combine_results([
            (phrase_results, 3.0),    # Highest weight for exact matches
            (keyword_results, 2.0),   # Good weight for keyword matches  
            (semantic_results, 1.5),  # Moderate weight for semantic
            (fuzzy_results, 1.0)      # Lowest weight for fuzzy
        ], top_k)
        
        return {"hits": combined_hits}
    
    def _phrase_search(self, query: str, top_k: int) -> List[dict]:
        """Exact phrase matching - best for titles and actor names"""
        
        body = {
            "size": top_k,
            "query": {
                "dis_max": {
                    "queries": [
                        # Exact phrase in title (highest priority)
                        {
                            "match_phrase": {
                                "title": {
                                    "query": query,
                                    "boost": 10.0
                                }
                            }
                        },
                        # Exact phrase in cast
                        {
                            "match_phrase": {
                                "cast": {
                                    "query": query,
                                    "boost": 8.0
                                }
                            }
                        },
                        # Exact phrase in director
                        {
                            "match_phrase": {
                                "director": {
                                    "query": query,
                                    "boost": 6.0
                                }
                            }
                        },
                        # Phrase in description
                        {
                            "match_phrase": {
                                "description": {
                                    "query": query,
                                    "boost": 2.0
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        try:
            response = self.client.search(index=self.index, body=body)
            return response.get("hits", {}).get("hits", [])
        except Exception:
            return []
    
    def _keyword_search(self, query: str, top_k: int) -> List[dict]:
        """Flexible keyword matching"""
        
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "type": "best_fields",
                    "fields": [
                        "title^8",
                        "cast^6", 
                        "director^5",
                        "listed_in_text^4",
                        "description^2",
                        "expanded_text^1.5"
                    ],
                    "minimum_should_match": "75%"
                }
            }
        }
        
        try:
            response = self.client.search(index=self.index, body=body)
            return response.get("hits", {}).get("hits", [])
        except Exception:
            return []
    
    def _semantic_search(self, query: str, top_k: int) -> List[dict]:
        """Semantic vector search"""
        
        try:
            vec = _qvec(query)
            body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "vector": {
                            "vector": vec,
                            "k": top_k
                        }
                    }
                }
            }
            
            response = self.client.search(index=self.index, body=body)
            return response.get("hits", {}).get("hits", [])
        except Exception:
            return []
    
    def _fuzzy_search(self, query: str, top_k: int) -> List[dict]:
        """Fuzzy matching for typos and partial matches"""
        
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "type": "best_fields", 
                    "fields": [
                        "title^4",
                        "cast^3",
                        "director^2", 
                        "listed_in_text^2"
                    ],
                    "fuzziness": "AUTO",
                    "minimum_should_match": "50%"
                }
            }
        }
        
        try:
            response = self.client.search(index=self.index, body=body)
            return response.get("hits", {}).get("hits", [])
        except Exception:
            return []
    
    def _combine_results(self, weighted_results: List[tuple], top_k: int) -> List[dict]:
        """Combine multiple result sets with weighted scoring"""
        
        doc_scores = {}  # doc_id -> (total_score, best_hit)
        
        for results, weight in weighted_results:
            for rank, hit in enumerate(results, 1):
                # Get document ID
                doc_id = self._get_doc_id(hit)
                if not doc_id:
                    continue
                    
                # Calculate weighted score (higher is better)
                original_score = hit.get("_score", 0)
                rank_score = 1.0 / rank  # Rank-based scoring
                final_score = (original_score * weight) + (rank_score * weight * 0.1)
                
                # Keep best scoring hit for each document
                if doc_id not in doc_scores or final_score > doc_scores[doc_id][0]:
                    doc_scores[doc_id] = (final_score, hit)
        
        # Sort by total score and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)
        return [hit for _, (score, hit) in sorted_docs[:top_k]]
    
    def _get_doc_id(self, hit: dict) -> Optional[str]:
        """Get consistent document ID"""
        source = hit.get("_source", {})
        return source.get("show_id") or hit.get("_id")


# Convenience function
def simple_search(query: str, top_k: int = 50, client: Optional[OpenSearch] = None, 
                 index: str = "netflix_assets_v6") -> Dict:
    """Perform simple robust search"""
    searcher = SimpleSearch(client=client, index=index)
    return searcher.search(query, top_k=top_k)


if __name__ == "__main__":
    # Test the simple search
    test_queries = [
        "Will Smith movies",           # Actor query
        "The Matrix",                  # Title query
        "romantic comedies 2020",      # Genre + year
        "Martin Scorsese films",       # Director query
        "vampire movies"               # Concept query
    ]
    
    print("Testing Simple Search System")
    print("=" * 50)
    
    searcher = SimpleSearch()
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = searcher.search(query, top_k=5)
        
        hits = result.get("hits", [])
        print(f"Found: {len(hits)} results")
        
        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            title = source.get("title", "Unknown")
            score = hit.get("_score", 0)
            print(f"  {i}. {title} (score: {score:.2f})")
        
        print("-" * 30)