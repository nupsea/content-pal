# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Content-Pal is a smart streaming content assistant that helps users find movies or TV shows to watch through a search-chat interface. The system is built around a sophisticated hybrid search architecture using OpenSearch, combining BM25 text search, semantic vector search (E5 embeddings), and cross-encoder re-ranking.

## Environment Setup

### Dependencies
The project uses Python 3.11 with pipenv for dependency management:

```bash
pipenv install
pipenv shell
```

Key dependencies include:
- opensearch-py for search engine integration
- sentence-transformers for embeddings (E5-small-v2 model)
- pandas for data processing
- transformers with PyTorch for ML models
- lightgbm for learning-to-rank features

### OpenSearch Infrastructure
Start the OpenSearch cluster with Docker Compose:

```bash
source .env  # or .envrc
docker-compose up -d
```

This creates:
- OpenSearch node at localhost:9200
- OpenSearch Dashboards at localhost:5601

Required environment variables (see .env_template):
- `OPENSEARCH_INITIAL_ADMIN_PASSWORD`
- `OS_USER`, `OS_PASS` for authentication
- `OPENAI_API_KEY` for LLM features

## Core Development Commands

### Data Indexing
Index Netflix dataset into OpenSearch:
```bash
cd src/search
python index_assets.py
```

### Search Evaluation
Evaluate search performance with ground truth data:
```bash
cd src/search
python eval_search_parallel.py \
  --index netflix_assets_v5 \
  --pairs ../../notebooks/ground_truth.json \
  --top_k 10 \
  --pool_top_k 80 \
  --workers 12 \
  --max_pairs 1000 \
  --use_cross_encoder
```

### Hyperparameter Tuning
Run grid search for optimal search parameters:
```bash
cd src/search
python sweep_hybrid.py
```

### Text Expansion (Doc2Query)
Generate expanded text for better search recall:
```bash
cd src/search
python expand_doc2query.py
```

## Architecture

### Search Pipeline
The hybrid search system consists of three main components:

1. **BM25 with Pseudo-Relevance Feedback (PRF)**: `hybrid_search.py:bm25_prf()`
   - Initial seed search to find expansion terms
   - Uses significant_text aggregation for query expansion
   - Multi-field search across title, cast, description, categories
   - Includes phrase and prefix matching variants

2. **Dense Vector Search**: `hybrid_search.py:knn_candidates()`
   - E5-small-v2 embeddings with cosine similarity
   - Query prefix: "query: " and document prefix: "passage: "
   - Vector dimension automatically detected from model

3. **Reciprocal Rank Fusion (RRF)**: `hybrid_search.py:rrf_fuse()`
   - Combines BM25 and vector search results
   - Configurable weights and rank constant
   - Deduplication by show_id

4. **Cross-Encoder Re-ranking**: `hybrid_search.py:rerank_topk()`
   - Optional final re-ranking step using cross-encoder models
   - Trained model path: `ce_netflix` (falls back to ms-marco baseline)

### Data Pipeline
- **Source Data**: Netflix titles CSV from Kaggle
- **Processing**: `index_assets.py:row_normalize()` handles data cleaning and normalization
- **Expansion**: Doc2query model generates synthetic queries for each document
- **Indexing**: Bulk indexing with optimized mapping for both text and vector search

### Evaluation Framework
- **Ground Truth**: `notebooks/ground_truth.json` contains curated query-document pairs
- **Metrics**: Hit Rate and Mean Reciprocal Rank (MRR) at k=10
- **Parallel Evaluation**: `eval_search_parallel.py` supports multi-threaded evaluation

## Key Configuration

Search behavior is controlled via environment variables:
- `HYB_BM25_SEED`, `HYB_BM25_FINAL`: BM25 result pool sizes
- `HYB_ANN_K`: Vector search candidates
- `HYB_RRF_K`: RRF rank constant
- `HYB_W_BM`, `HYB_W_ANN`: Fusion weights
- `HYB_EXP_BOOST`: Query expansion boost factor

## Directory Structure

- `src/search/`: Main search implementation and evaluation scripts
- `src/archive/`: Legacy learning-to-rank experiments
- `data/`: Netflix dataset files (original, cleaned, expanded)
- `notebooks/`: Analysis notebooks and ground truth data
- `checkpoints/`: Model checkpoints from training experiments

## Development Workflow

1. Ensure OpenSearch cluster is running
2. Index data: `python src/search/index_assets.py`
3. Test search: Use evaluation scripts or notebooks for experimentation
4. Tune parameters: Use sweep scripts for optimization
5. Evaluate: Run parallel evaluation against ground truth