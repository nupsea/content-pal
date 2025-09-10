# Content Pal
A smart streaming content assistant to help users find what movies or tv-shows to watch and obtain adhoc recommendations from a search-chat interface.  

## Setup

### Installation 

Prerequisites: Python 3.11 or higher
Package Mangaer: pipenv (in case of any others the instructions need to change accordingly)

Install the required libraries.

```
pipenv install openai scikit-learn pandas flask minsearch opensearch-py sentence-transformers lightgbm scikit-learn   
pipenv install --dev tqdm ipywidgets python-dotenv  transformers[torch] datasets

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


## Usage

### CLI with Flask

To start the Flask server, run the following command:

```zsh

pipenv run python -m src.modules.workflow.app
```

From a new terminal, you can interact with the API using curl or any API client like Postman.
```zsh
❯ curl -X POST \
-H "Content-Type: application/json" \
-d '{"query": "Mind bending shows or movies"}' \
http://127.0.0.1:5000/recommend
"{\n  \"catalog_recommendations\": [\n    \"Synchronic (2020): Two paramedics begin to question their realities after coming across several bizarre deaths linked to a new narcotic with mind-bending effects.\",\n    \"ANIMA (2019): In a short musical film directed by Paul Thomas Anderson, Thom Yorke of Radiohead stars in a mind-bending visual piece. Best played loud.\",\n    \"Black Mirror: Bandersnatch (2018): In 1984, a young programmer begins to question reality as he adapts a dark fantasy novel into a video game. A mind-bending tale with multiple endings.\",\n    \"Maniac (2018): Two struggling strangers connect during a mind-bending pharmaceutical trial involving a doctor with mother issues and an emotionally complex computer.\",\n    \"Dark (2020): A missing child sets four families on a frantic hunt for answers as they unearth a mind-bending mystery that spans three generations.\"\n  ]\n}"
```

Alternatively, you could use the test script provided in `src/modules/workflow/test.py`.

```zsh
pipenv run python -m src.modules.workflow.test
Loading .env environment variables...

User Query: Feel good rom com movies recent
Response: {
  "catalog_recommendations": [
    "Good on Paper (2021): After years of putting her career first, a stand-up comic meets a guy who seems perfect: smart, nice, successful... and possibly too good to be true.",
    "Feel Good (2021): Stand-up comic Mae Martin navigates a passionate, messy new relationship with her girlfriend, George, while dealing with the challenges of sobriety.",
    "Feel the Beat (2020): After blowing a Broadway audition, a self-centered dancer reluctantly returns home and agrees to coach a squad of young misfits for a big competition.",
    "Tell Me When (2021): Workaholic Will puts his humdrum life in LA on hold to fulfill his grandpa's last wish: visiting Mexico City's most iconic sights and falling in love.",
    "Good Luck Chuck (2007): Every time Chuck breaks up with a girlfriend, she ends up engaged to her next boyfriend. Soon, women are dating Chuck in hopes of meeting Mr. Right."
  ]
}
```

## Monitoring


