#!/usr/bin/env python3
"""
Generic media content search interface supporting multiple backends.
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import backends conditionally
try:
    import minsearch
    HAS_MINSEARCH = True
except ImportError:
    HAS_MINSEARCH = False

try:
    from opensearchpy import OpenSearch
    from sentence_transformers import SentenceTransformer
    HAS_OPENSEARCH = True
except ImportError:
    HAS_OPENSEARCH = False


@dataclass
class SearchResult:
    """Standardized search result"""
    id: str
    title: str
    score: float
    content_type: str
    metadata: Dict[str, str]  # Flexible metadata storage


@dataclass
class SearchConfig:
    """Search configuration"""
    boost_weights: Dict[str, float]
    max_results: int = 50


class SearchBackend(ABC):
    """Abstract base class for search backends"""
    
    @abstractmethod
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents"""
        pass
    
    @abstractmethod
    def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search documents"""
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Get backend name"""
        pass


class MinSearchBackend(SearchBackend):
    """MinSearch backend implementation"""
    
    def __init__(self, text_fields: List[str], keyword_fields: List[str]):
        if not HAS_MINSEARCH:
            raise ImportError("minsearch not available")
        
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.index = None
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents with MinSearch"""
        print(f"Indexing {len(documents)} documents with MinSearch...")
        
        self.index = minsearch.Index(
            text_fields=self.text_fields,
            keyword_fields=self.keyword_fields
        )
        self.index.fit(documents)
        print("MinSearch indexing complete")
    
    def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search with MinSearch"""
        if not self.index:
            raise RuntimeError("Index not created. Call index_documents first.")
        
        results = self.index.search(
            query=query,
            boost_dict=config.boost_weights,
            num_results=config.max_results
        )
        
        # Convert to standardized format
        search_results = []
        for result in results:
            # Separate core fields from metadata
            core_fields = {'id', 'title', 'type'}
            metadata = {k: v for k, v in result.items() if k not in core_fields}
            
            search_results.append(SearchResult(
                id=result['id'],
                title=result['title'],
                score=1.0,  # MinSearch doesn't provide scores
                content_type=result.get('type', ''),
                metadata=metadata
            ))
        
        return search_results
    
    def get_backend_name(self) -> str:
        return "MinSearch"


class OpenSearchBackend(SearchBackend):
    """OpenSearch backend implementation with hybrid search support"""
    
    def __init__(self, text_fields: List[str], keyword_fields: List[str], 
                 index_name: str = None):
        if not HAS_OPENSEARCH:
            raise ImportError("OpenSearch not available")
        
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.index_name = index_name or os.getenv("OS_INDEX", "netflix_assets_v6")
        self.client = self._create_client()
        self.embedding_model = None
        self._hybrid_available = self._check_hybrid_search()
    
    def _create_client(self) -> OpenSearch:
        """Create OpenSearch client using existing infrastructure"""
        try:
            # Try to use existing client creation from index_assets
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "search"))
            from index_assets import make_client
            return make_client()
        except ImportError:
            # Fallback to basic client
            return OpenSearch(
                os.getenv("OS_URL", "https://localhost:9200"),
                http_auth=(os.getenv("OS_USER", "admin"), os.getenv("OS_PASS", "admin")),
                verify_certs=False,
                ssl_show_warn=False,
                timeout=60
            )
    
    def _check_hybrid_search(self) -> bool:
        """Check if hybrid search is available"""
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "search"))
            from hybrid_search import hybrid_search
            return True
        except ImportError:
            return False
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if not self.embedding_model:
            model_name = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
            self.embedding_model = SentenceTransformer(model_name)
        return self.embedding_model
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents with OpenSearch"""
        print(f"Indexing {len(documents)} documents with OpenSearch...")
        
        # Create dynamic mapping based on fields
        properties = {
            "id": {"type": "keyword"},
            "title": {"type": "text"},
            "type": {"type": "keyword"},
            "vector": {
                "type": "knn_vector", 
                "dimension": 1024,
                "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"}
            }
        }
        
        # Add text fields
        for field in self.text_fields:
            if field not in properties:
                properties[field] = {"type": "text"}
        
        # Add keyword fields  
        for field in self.keyword_fields:
            if field not in properties:
                properties[field] = {"type": "keyword"}
        
        mapping = {
            "settings": {"index": {"knn": True, "refresh_interval": "1s"}},
            "mappings": {"properties": properties}
        }
        
        # Recreate index
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
        except Exception:
            pass
        
        self.client.indices.create(index=self.index_name, body=mapping)
        
        # Index with embeddings
        model = self._get_embedding_model()
        bulk_data = []
        
        for doc in documents:
            # Create embedding text from main text fields
            embed_text = " ".join(str(doc.get(field, '')) for field in self.text_fields)
            vector = model.encode(embed_text, normalize_embeddings=True)
            
            index_doc = {**doc, "vector": vector.tolist()}
            bulk_data.extend([
                {"index": {"_index": self.index_name, "_id": doc['id']}},
                index_doc
            ])
        
        from opensearchpy.helpers import bulk
        bulk(self.client, bulk_data, chunk_size=500)
        self.client.indices.refresh(index=self.index_name)
        print("OpenSearch indexing complete")
    
    def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search with OpenSearch"""
        if not self.client.indices.exists(index=self.index_name):
            raise RuntimeError("Index not found. Call index_documents first.")
        
        # Build query with boost weights
        fields = [f"{field}^{config.boost_weights.get(field, 1.0)}" 
                 for field in self.text_fields]
        
        search_body = {
            "size": config.max_results,
            "query": {
                "multi_match": {
                    "query": query,
                    "type": "best_fields", 
                    "fields": fields,
                    "minimum_should_match": "75%"
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        # Convert to standardized format
        search_results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            core_fields = {'id', 'title', 'type', 'vector'}
            metadata = {k: v for k, v in source.items() if k not in core_fields}
            
            search_results.append(SearchResult(
                id=source['id'],
                title=source['title'],
                score=hit['_score'],
                content_type=source.get('type', ''),
                metadata=metadata
            ))
        
        return search_results
    
    def get_backend_name(self) -> str:
        return "OpenSearch"


class ContentSearchSystem:
    """Generic content search system with switchable backends"""
    
    def __init__(self, backend_type: str = "minsearch", 
                 text_fields: List[str] = None, keyword_fields: List[str] = None):
        """Initialize with specified backend and field configuration"""
        
        # Default fields for media content
        if not text_fields:
            text_fields = ['title', 'description', 'cast', 'director', 'listed_in']
        if not keyword_fields:
            keyword_fields = ['id', 'type', 'release_year']
        
        self.backend_type = backend_type.lower()
        
        if self.backend_type == "minsearch":
            self.backend = MinSearchBackend(text_fields, keyword_fields)
        elif self.backend_type == "opensearch":
            self.backend = OpenSearchBackend(text_fields, keyword_fields)
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
        
        self.indexed = False
    
    def load_data(self, csv_path: str, id_field: str = 'show_id', 
                  title_field: str = 'title', type_field: str = 'type') -> List[Dict]:
        """Load data from CSV"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, encoding='latin-1').fillna('')
        
        documents = []
        for _, row in df.iterrows():
            # Ensure required fields exist
            doc = row.to_dict()
            doc['id'] = doc.pop(id_field, row.iloc[0])
            doc['title'] = doc.get(title_field, '')
            doc['type'] = doc.get(type_field, '')
            documents.append(doc)
        
        print(f"Loaded {len(documents)} items")
        return documents
    
    def index_data(self, csv_path: str = None, documents: List[Dict] = None, **kwargs) -> None:
        """Index data from CSV or document list"""
        if documents is None:
            if csv_path is None:
                csv_path = 'data/netflix_titles_cleaned.csv'
            documents = self.load_data(csv_path, **kwargs)
        
        self.backend.index_documents(documents)
        self.indexed = True
    
    def search(self, query: str, config: SearchConfig = None) -> List[SearchResult]:
        """Search content"""
        if not self.indexed:
            raise RuntimeError("Data not indexed. Call index_data() first.")
        
        if not config:
            # Optimized default config from our evaluation
            config = SearchConfig(
                boost_weights={'title': 2.0, 'cast': 4.0, 'director': 2.0, 
                             'listed_in': 1.0, 'description': 0.5},
                max_results=50
            )
        
        return self.backend.search(query, config)
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'backend_type': self.backend_type,
            'backend_name': self.backend.get_backend_name(),
            'indexed': self.indexed
        }


# Convenience functions
def create_content_search(backend_type: str = "minsearch", **kwargs) -> ContentSearchSystem:
    """Create content search system"""
    return ContentSearchSystem(backend_type=backend_type, **kwargs)


if __name__ == "__main__":
    # Test both backends
    test_queries = ["Will Smith movies", "The Matrix", "romantic comedies"]
    
    for backend in ["minsearch", "opensearch"]:
        print(f"\n{'='*50}")
        print(f"Testing {backend.upper()}")
        print("=" * 50)
        
        try:
            search = create_content_search(backend)
            search.index_data()
            
            for query in test_queries:
                results = search.search(query)
                print(f"\n'{query}': {len(results)} results")
                for i, r in enumerate(results[:2], 1):
                    print(f"  {i}. {r.title}")
        
        except Exception as e:
            print(f"Error: {e}")