"""
Core search data structures and interfaces
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Standardized search result"""
    id: str
    title: str
    score: float
    content_type: str
    metadata: Dict[str, Any]


@dataclass
class SearchConfig:
    """Search configuration"""
    boost_weights: Dict[str, float]
    max_results: int = 50
    use_reranking: bool = False
    use_hybrid: bool = True


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


class ContentSearchSystem:
    """Generic content search system with switchable backends"""
    
    def __init__(self, backend_type: str = "minsearch", 
                 text_fields: Optional[List[str]] = None, keyword_fields: Optional[List[str]] = None):
        """Initialize with specified backend and field configuration"""
        
        # Default fields for media content
        if not text_fields:
            text_fields = ['title', 'description', 'cast', 'director', 'listed_in']
        if not keyword_fields:
            keyword_fields = ['id', 'type', 'release_year']
        
        self.backend_type = backend_type.lower()
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        
        # Import and create backend
        if self.backend_type == "minsearch":
            from .backends import MinSearchBackend
            self.backend = MinSearchBackend(text_fields, keyword_fields)
        elif self.backend_type == "opensearch":
            from .backends import OpenSearchBackend
            self.backend = OpenSearchBackend(text_fields, keyword_fields)
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
        
        self.indexed = False
    
    def load_data(self, csv_path: str, id_field: str = 'show_id', 
                  title_field: str = 'title', type_field: str = 'type') -> List[Dict]:
        """Load data from CSV"""
        df = pd.read_csv(csv_path, encoding='latin-1').fillna('')
        
        documents = []
        for _, row in df.iterrows():
            doc = row.to_dict()
            doc['id'] = doc.pop(id_field, row.iloc[0])
            doc['title'] = doc.get(title_field, '')
            doc['type'] = doc.get(type_field, '')
            documents.append(doc)
        
        return documents

    def index_data(self, csv_path: Optional[str] = None, documents: Optional[List[Dict]] = None, **kwargs) -> None:
        """Index data from CSV or document list"""
        if documents is None:
            if csv_path is None:
                csv_path = 'data/netflix_titles_cleaned.csv'
            documents = self.load_data(csv_path, **kwargs)
        
        self.backend.index_documents(documents)
        self.indexed = True
    
    def search(self, query: str, config: Optional[SearchConfig] = None) -> List[SearchResult]:
        """Search content"""
        if not self.indexed:
            raise RuntimeError("Data not indexed. Call index_data() first.")
        
        if not config:
            # Optimized default config
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
            'indexed': self.indexed,
            'text_fields': self.text_fields,
            'keyword_fields': self.keyword_fields
        }


# Convenience function
def create_content_search(backend_type: str = "minsearch", **kwargs) -> ContentSearchSystem:
    """Create content search system"""
    return ContentSearchSystem(backend_type=backend_type, **kwargs)