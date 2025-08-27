"""
Search backend implementations
"""

import os
from typing import Dict, List, Optional
from .core import SearchBackend, SearchResult, SearchConfig

import minsearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer


class MinSearchBackend(SearchBackend):
    """MinSearch backend implementation"""
    
    def __init__(self, text_fields: List[str], keyword_fields: List[str]):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.index = None
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents with MinSearch"""
        self.index = minsearch.Index(
            text_fields=self.text_fields,
            keyword_fields=self.keyword_fields
        )
        self.index.fit(documents)
    
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
                 index_name: Optional[str] = None):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.index_name = index_name or os.getenv("OS_INDEX", "netflix_assets_v6")
        self.client = self._create_client()
        self.embedding_model = None
        self.index_exists = self.client.indices.exists(index=self.index_name)
    
    def _create_client(self) -> OpenSearch:
        """Create OpenSearch client"""
        os_url = os.getenv("OS_URL", "https://localhost:9200")
        os_user = os.getenv("OS_USER", "admin")
        os_pass = os.getenv("OS_PASS", "admin")
        os_verify = os.getenv("OS_VERIFY", "false").lower() == "true"
        
        auth = (os_user, os_pass) if (os_user and os_pass) else None
        
        if os_url.startswith("https://"):
            return OpenSearch(
                os_url,
                http_auth=auth,
                verify_certs=os_verify,
                ssl_assert_hostname=os_verify,
                ssl_show_warn=os_verify,
                connection_class=RequestsHttpConnection,
                timeout=60, max_retries=3, retry_on_timeout=True,
            )
        return OpenSearch(os_url, http_auth=auth)
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if not self.embedding_model:
            model_name = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
            self.embedding_model = SentenceTransformer(model_name)
        return self.embedding_model
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents with OpenSearch"""
        try:
            from .indexer import OpenSearchIndexer
            indexer = OpenSearchIndexer(self.client, self.index_name)
            indexer.index_documents(documents)
        except ImportError:
            self._basic_index_documents(documents)
    
    def _basic_index_documents(self, documents: List[Dict]) -> None:
        """Basic document indexing without embeddings"""
        # Create basic mapping
        properties = {
            "show_id": {"type": "keyword"},
            "id": {"type": "keyword"},
            "title": {"type": "text"},
            "type": {"type": "keyword"},
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
            "settings": {"index": {"refresh_interval": "1s"}},
            "mappings": {"properties": properties}
        }
        
        # Recreate index
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        
        self.client.indices.create(index=self.index_name, body=mapping)
        
        # Index documents
        from opensearchpy.helpers import bulk
        bulk_data = []
        
        for doc in documents:
            # Ensure we have an ID
            doc_id = doc.get('id') or doc.get('show_id') or str(hash(doc.get('title', '')))
            bulk_data.extend([
                {"index": {"_index": self.index_name, "_id": doc_id}},
                doc
            ])
        
        bulk(self.client, bulk_data, chunk_size=500)
        self.client.indices.refresh(index=self.index_name)
    
    def search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Search with OpenSearch"""
        if not self.client.indices.exists(index=self.index_name):
            raise RuntimeError("Index not found. Call index_documents first.")
        
        # Try hybrid search first if available
        if config.use_hybrid:
            try:
                from .hybrid import hybrid_search
                hits = hybrid_search(self.client, self.index_name, query, config.max_results)
            except ImportError:
                hits = self._basic_search(query, config)
        else:
            hits = self._basic_search(query, config)
        
        # Convert to standardized format
        search_results = []
        for hit in hits:
            source = hit.get('_source', {})
            # Use show_id as primary ID for Netflix data
            doc_id = source.get('show_id') or source.get('id') or hit.get('_id', '')
            title = source.get('title', '')
            content_type = source.get('type', '')
            score = hit.get('_score', 0.0)
            
            # Separate core fields from metadata
            core_fields = {'show_id', 'id', 'title', 'type', 'vector'}
            metadata = {k: v for k, v in source.items() if k not in core_fields}
            
            search_results.append(SearchResult(
                id=doc_id,
                title=title,
                score=score,
                content_type=content_type,
                metadata=metadata
            ))
        
        return search_results
    
    def _basic_search(self, query: str, config: SearchConfig) -> List[dict]:
        """Basic search implementation"""
        # Map fields to actual schema
        field_mappings = {
            'title': 'title',
            'cast': 'cast',
            'director': 'director',
            'listed_in': 'listed_in',
            'description': 'description'
        }
        
        fields = []
        for field in self.text_fields:
            mapped_field = field_mappings.get(field, field)
            boost = config.boost_weights.get(field, 1.0)
            fields.append(f"{mapped_field}^{boost}")
        
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
        return response.get('hits', {}).get('hits', [])
    
    def get_backend_name(self) -> str:
        return "OpenSearch"