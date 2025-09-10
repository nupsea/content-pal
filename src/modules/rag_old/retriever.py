"""
Adaptive retriever that uses query classification to optimize search strategy
"""

from typing import Dict, List, Any, Optional
from ..search import ContentSearchSystem, SearchConfig
from .query_classifier import QueryClassifier, IntentType


class AdaptiveRetriever:
    """Adaptive retriever that adjusts search strategy based on query intent"""

    def __init__(self, search_system: Optional[ContentSearchSystem] = None, backend_type: Optional[str] = None, **kwargs):
        """Initialize with search system or create one"""
        if search_system:
            self.search_system = search_system
        else:
            if backend_type:
                self.search_system = ContentSearchSystem(backend_type=backend_type, **kwargs)
            else:
                try:
                    self.search_system = ContentSearchSystem(backend_type="opensearch", **kwargs)
                except ImportError:
                    self.search_system = ContentSearchSystem(backend_type="minsearch", **kwargs)
        
        self.classifier = QueryClassifier()
        
        # Strategy-specific configurations
        self.strategy_configs = {
            IntentType.TITLE_SEARCH: SearchConfig(
                boost_weights={'title': 8.0, 'cast': 1.0, 'director': 0.5, 
                             'listed_in': 0.5, 'description': 0.3},
                max_results=50,
                use_hybrid=True
            ),
            IntentType.ACTOR_SEARCH: SearchConfig(
                boost_weights={'title': 2.0, 'cast': 8.0, 'director': 1.0, 
                             'listed_in': 1.0, 'description': 0.5},
                max_results=50,
                use_hybrid=True
            ),
            IntentType.GENRE_SEARCH: SearchConfig(
                boost_weights={'title': 1.5, 'cast': 1.0, 'director': 1.0, 
                             'listed_in': 6.0, 'description': 2.0},
                max_results=50,
                use_hybrid=True
            ),
            IntentType.DIRECTOR_SEARCH: SearchConfig(
                boost_weights={'title': 2.0, 'cast': 1.0, 'director': 8.0, 
                             'listed_in': 1.0, 'description': 1.0},
                max_results=50,
                use_hybrid=True
            ),
            IntentType.CONTENT_SEARCH: SearchConfig(
                boost_weights={'title': 2.0, 'cast': 1.5, 'director': 1.0, 
                             'listed_in': 3.0, 'description': 6.0},
                max_results=50,
                use_hybrid=True
            ),
            IntentType.YEAR_SEARCH: SearchConfig(
                boost_weights={'title': 3.0, 'cast': 2.0, 'director': 2.0, 
                             'listed_in': 2.0, 'description': 1.0},
                max_results=50,
                use_hybrid=False  # Year searches work better with keyword matching
            ),
            IntentType.COMBINED_SEARCH: SearchConfig(
                boost_weights={'title': 4.0, 'cast': 3.0, 'director': 2.0, 
                             'listed_in': 2.5, 'description': 2.0},
                max_results=50,
                use_hybrid=True
            )
        }
        
        # Default fallback config
        self.default_config = SearchConfig(
            boost_weights={'title': 3.0, 'cast': 2.5, 'director': 2.0, 
                         'listed_in': 2.0, 'description': 1.5},
            max_results=50,
            use_hybrid=True
        )
    
    def index_data(self, **kwargs):
        """Index data using the underlying search system"""
        self.search_system.index_data(**kwargs)
    
    def retrieve(self, query: str, top_k: int = 50, use_reranking: bool = False, 
                 custom_config: Optional[SearchConfig] = None) -> Dict[str, Any]:
        """
        Retrieve documents using adaptive strategy
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use cross-encoder reranking
            custom_config: Override automatic config selection
            
        Returns:
            Dictionary with hits, intent info, and metadata
        """
        # Classify query intent
        intent = self.classifier.classify_query(query)
        
        # Get appropriate search configuration
        if custom_config:
            config = custom_config
            strategy = "custom"
        else:
            config = self.strategy_configs.get(intent.intent_type, self.default_config)
            strategy = intent.intent_type.value
        
        # Update config with parameters
        config.max_results = top_k
        config.use_reranking = use_reranking
        
        # Execute search
        try:
            results = self.search_system.search(query, config)
            
            # Convert to hits format for compatibility
            hits = []
            for result in results:
                hit = {
                    '_id': result.id,
                    '_score': result.score,
                    '_source': {
                        'show_id': result.id,
                        'title': result.title,
                        'type': result.content_type,
                        **result.metadata
                    }
                }
                hits.append(hit)
            
            # Apply reranking if requested and available
            if use_reranking and hits:
                from ..search.hybrid import rerank_topk
                hits = rerank_topk(query, hits, k=min(top_k, 20))
            
            return {
                'hits': hits,
                'query_intent': intent,
                'strategy_used': strategy,
                'total_results': len(hits),
                'reranked': use_reranking,
                'backend_info': self.search_system.get_info()
            }
            
        except Exception as e:
            raise e
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the retrieval system"""
        return {
            'search_backend': self.search_system.get_info(),
            'classifier': 'QueryClassifier',
            'strategies_available': list(self.strategy_configs.keys()),
            'supports_reranking': True,  # Attempt reranking if deps available
        }