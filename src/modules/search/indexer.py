"""
OpenSearch indexing functionality
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Iterator

import pandas as pd
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from sentence_transformers import SentenceTransformer

# Date patterns for parsing
_DATE_PATTERNS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")
_RE_WS = re.compile(r"\s+")


class OpenSearchIndexer:
    """Handles OpenSearch indexing with embeddings and proper mapping"""
    
    def __init__(self, client: OpenSearch, index_name: str):
        self.client = client
        self.index_name = index_name
        self.embedding_model = None
        self.vector_dim = None
    
    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if not self.embedding_model:
            model_name = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
            self.embedding_model = SentenceTransformer(model_name)
            
            # Determine vector dimension
            probe = self.embedding_model.encode(["probe"], normalize_embeddings=True)
            self.vector_dim = int(probe.shape[1]) if hasattr(probe, "shape") else len(probe[0])
        
        return self.embedding_model, self.vector_dim
    
    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and normalize text"""
        if text is None:
            return None
        text = str(text).strip()
        return text or None
    
    def _split_csv(self, text: Optional[str]) -> List[str]:
        """Split CSV field into list"""
        text = self._clean_text(text)
        return [p.strip() for p in text.split(",")] if text else []
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse date to ISO format"""
        if not date_str:
            return None
        date_str = str(date_str).strip()
        for fmt in _DATE_PATTERNS:
            try:
                return datetime.strptime(date_str, fmt).date().isoformat()
            except Exception:
                continue
        return None
    
    def _normalize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize document for indexing"""
        listed_in = self._split_csv(doc.get("listed_in"))
        cast_list = self._split_csv(doc.get("cast"))
        
        # Parse release year
        ry = self._clean_text(doc.get("release_year"))
        release_year = int(ry) if ry and str(ry).isdigit() else None
        
        # Parse date
        raw_date = self._clean_text(doc.get("date_added"))
        iso_date = self._parse_date(raw_date)
        
        normalized = {
            "show_id": self._clean_text(doc.get("show_id")) or self._clean_text(doc.get("id")),
            "type": self._clean_text(doc.get("type")),
            "title": self._clean_text(doc.get("title")),
            "director": self._clean_text(doc.get("director")),
            "cast": self._clean_text(doc.get("cast")),
            "cast_list": cast_list,
            "country": self._clean_text(doc.get("country")),
            "date_added_raw": raw_date,
            "release_year": release_year,
            "rating": self._clean_text(doc.get("rating")),
            "duration": self._clean_text(doc.get("duration")),
            "listed_in": listed_in,
            "listed_in_text": ", ".join(listed_in) if listed_in else None,
            "description": self._clean_text(doc.get("description")),
            "expanded_text": self._clean_text(doc.get("expanded_text")),
        }
        
        # Add parsed date if available
        if iso_date:
            normalized["date_added"] = iso_date
        
        # Remove empty fields
        normalized = {k: v for k, v in normalized.items() if v not in (None, "", [], {})}
        
        # Ensure stable ID
        doc_id = normalized.get("show_id")
        if not doc_id:
            slug = _RE_WS.sub("", (normalized.get("title") or "").lower())
            doc_id = f"{slug}_{release_year if release_year else 'na'}_gen"
            normalized["show_id"] = doc_id
        
        normalized["_id"] = doc_id
        return normalized
    
    def _text_for_embedding(self, doc: Dict[str, Any]) -> str:
        """Create text for embedding"""
        parts = []
        for field in ("title", "description", "director", "type", "country", "rating"):
            value = doc.get(field)
            if value:
                parts.append(str(value))
        
        if doc.get("cast_list"):
            parts.append(", ".join(doc["cast_list"]))
        if doc.get("listed_in"):
            parts.append(", ".join(doc["listed_in"]))
        
        return " | ".join(parts)
    
    def _create_index_mapping(self, vector_dim: int) -> Dict[str, Any]:
        """Create index mapping with all necessary fields"""
        return {
            "settings": {
                "index": {"knn": True, "refresh_interval": "1s"},
                "analysis": {
                    "analyzer": {
                        "english_desc": {
                            "tokenizer": "standard", 
                            "filter": ["lowercase", "porter_stem", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "show_id": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                    "director": {"type": "text"},
                    "cast": {"type": "text"},
                    "cast_list": {"type": "keyword"},
                    "country": {"type": "keyword"},
                    "date_added": {"type": "date"},
                    "date_added_raw": {"type": "keyword"},
                    "release_year": {"type": "integer"},
                    "rating": {"type": "keyword"},
                    "duration": {"type": "keyword"},
                    "listed_in": {"type": "keyword"},
                    "listed_in_text": {"type": "text"},
                    "description": {"type": "text", "analyzer": "english_desc"},
                    "expanded_text": {"type": "text"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": vector_dim,
                        "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"},
                    },
                }
            }
        }
    
    def _recreate_index(self, vector_dim: int) -> None:
        """Recreate index with proper mapping"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        
        mapping = self._create_index_mapping(vector_dim)
        self.client.indices.create(index=self.index_name, body=mapping)
        self.client.cluster.health(index=self.index_name)
        self.client.indices.refresh(index=self.index_name)
    
    def _generate_bulk_actions(self, documents: List[Dict]) -> Iterator[Dict[str, Any]]:
        """Generate bulk indexing actions with embeddings"""
        model, _ = self._get_embedding_model()
        if not model:
            # Index without embeddings
            for doc in documents:
                normalized = self._normalize_document(doc)
                doc_id = normalized.pop("_id")
                yield {"_op_type": "index", "_index": self.index_name, "_id": doc_id, **normalized}
            return
        
        # Normalize all documents
        normalized_docs = [self._normalize_document(doc) for doc in documents]
        
        # Create embedding texts
        embed_texts = [self._text_for_embedding(doc) for doc in normalized_docs]
        
        # Process in batches
        batch_size = int(os.getenv("EMBED_BATCH", "64"))
        for i in range(0, len(normalized_docs), batch_size):
            batch_docs = normalized_docs[i:i+batch_size]
            batch_texts = embed_texts[i:i+batch_size]
            
            # Generate embeddings for batch
            vectors = model.encode(batch_texts, normalize_embeddings=True, 
                                 batch_size=batch_size, show_progress_bar=False)
            vectors_list = vectors.tolist() if hasattr(vectors, "tolist") else vectors
            
            # Yield actions
            for doc, vector in zip(batch_docs, vectors_list):
                doc_id = doc.pop("_id")
                action = {"_op_type": "index", "_index": self.index_name, "_id": doc_id, **doc}
                if vector:
                    action["vector"] = vector if isinstance(vector, list) else list(vector)
                yield action
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents with proper mapping and embeddings"""
        model, vector_dim = self._get_embedding_model()
        
        # Use default dimension if no model available
        if vector_dim is None:
            vector_dim = 1024
        
        # Recreate index
        self._recreate_index(vector_dim)
        
        # Bulk index documents
        bulk(
            self.client, 
            self._generate_bulk_actions(documents), 
            stats_only=True, 
            raise_on_error=True
        )
        
        # Refresh index
        self.client.indices.refresh(index=self.index_name)