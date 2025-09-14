"""
Configuration for Streamlit UI when running in Docker
"""
import os

# Determine if running in Docker
IN_DOCKER = os.path.exists('/.dockerenv')

# API Base URL - adjust based on environment
if IN_DOCKER:
    # When running in Docker, connect to 'app' service
    BASE_URL = "http://app:5001"
else:
    # When running locally, connect to localhost
    BASE_URL = "http://localhost:5001"

# Database configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'database': os.getenv('POSTGRES_DB', 'content_pal'), 
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

# Data files
RANDOM_QUERIES_FILE = "./data/ground_truth_retrieval.csv"