import os
import pandas as pd

from src.modules.search.aware_pts import SchemaAwareSemanticSearch

DATA_PATH = os.getenv("DATA_PATH", "data/netflix_titles_enriched_full.csv")


def load_index(data_path: str = DATA_PATH):
    """Load or create MinSearch index from CSV data"""

    search_system = SchemaAwareSemanticSearch(backend_type="minsearch")
    search_system.index_data(csv_path=DATA_PATH)

    return search_system
    
