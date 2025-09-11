# Content Pal
A smart streaming content assistant to help users find what movies or tv-shows to watch and obtain adhoc recommendations from a search-chat interface.  

## Setup

### Installation 

Prerequisites: Python 3.11 or higher
Package Mangaer: pipenv (in case of any others the instructions need to change accordingly)

Install the required libraries.

```
pipenv install openai scikit-learn pandas flask minsearch opensearch-py sentence-transformers lightgbm scikit-learn psycopg2-binary  sqlalchemy sqlalchemy-utils python-multipart flask-cors gunicorn
```
pipenv install --dev tqdm ipywidgets python-dotenv transformers[torch] datasets

# TODO check streamlit

```


### Prep Internal Search Data

Data obtained from Kaggle Netflix Movies and TV shows.

### Search Engines
Index Search data into Opensearch or minsearch a mini search engine implementation.

#### Minsearch
Minsearch is a lightweight, in-memory search engine designed for quick indexing and retrieval of documents. It is ideal for smaller datasets or when rapid prototyping is needed.
Install Minsearch
```sh
pipenv install minsearch
```

## Running it with Docker

The entire application can be run using Docker and Docker-Compose.
Make sure you have Docker and Docker-Compose installed on your machine.

If you need to change some environment variables, you can do so in the `.env` file and
correspondingly build DockerFile. 

```zsh
docker build -t content-pal .

source .env && 
docker run -it --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e DATA_PATH="data/netflix_titles_enriched_full.csv" \
  -p 5001:5000 \
  content-pal
```

### Testing with app
You can test the API using curl or any API client like Postman.
```zsh
curl -X POST "http://localhost:5001/recommend" -H "Content-Type: application/json" -d '{"query": "Recommend me some sci-fi movies"}'
```
Or Alternatively, you can use the provided `test.py` script to test the API.
```zsh
pipenv run python -m src.modules.workflow.test -p 5001
```


#### Opensearch (Optional)

Search Engine Start with Docker-Compose
```sh
source .envrc
docker-compose up -d
```

You should see something like the below : 

[+] Running 3/3
 ✔ Network content-pal_default  Created                                                                                                                                                                                                     0.0s 
 ✔ Container opensearch         Healthy                                                                                                                                                                                                    10.6s 
 ✔ Container os-dashboards      Started                                                                                                                                                                                                    10.7s 
❯ 


### Execution steps


## Ingestion and Indexing

The ingestion script is part of `src/modules/workflow/ingest.py` for the simple minsearch based ingestion.

Since we use an in-memory database, minsearch, as our knowledge base, we run the ingestion script at the startup of the application.
It's executed inside `src/modules/workflow/rag.py` when we import it.


The ingestion script reads the data from the CSV file, processes it, and indexes it into the available search engines.



## Evaluation

### Retrieval Evaluation

Initially evaluated the search results with basic search engines (minsearch and opensearch) and then used both keyword based and vector based search using BM25 and ANN (Approximate Nearest Neighbors) based search.
Refer `notebooks/analysis.ipynb` for details.


Later in order to improve the search quality, used pretraining models and cross-encoders to rerank the results. Also, used different strategies and combinations to acheive better results.

```zsh
❯ pipenv run python src/run_clean_evaluation.py
Loading .env environment variables...
CLEAN COMPREHENSIVE EVALUATION
============================================================
[OK] Loaded ground truth: 1000 unique assets
[OK] Created evaluation subset: 200 assets
..


SchemaAwareSemanticSearch:
  Total Queries: 500
  Successful Queries: 500
  Hit Rate @ 10: 38.0%
  MRR @ 10: 26.5%
  Avg Query Time: 4588.4ms

SimpleEffectiveSearch:
  Total Queries: 500
  Successful Queries: 500
  Hit Rate @ 10: 32.6%
  MRR @ 10: 22.0%
  Avg Query Time: 1164.0ms

UltraBoostSearch:
  Total Queries: 500
  Successful Queries: 500
  Hit Rate @ 10: 34.6%
  MRR @ 10: 23.5%
  Avg Query Time: 4016.9ms

```

### RAG Flow Evaluation

Used LLM-as-a-Judge metric to evaluate the content results.

Among 100 sampled queries, obtained:

* RELEVANT:         48%
* PARTIAL_RELEVANT: 39%
* NOT_RELEVANT:     13% 

Refer `notebooks/rag_flow.ipynb` for details.



## Monitoring


