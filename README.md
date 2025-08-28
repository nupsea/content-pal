# Content Pal
A smart streaming content assistant to help users find what movies or tv-shows to watch and obtain adhoc recommendations from a search-chat interface.  

## Setup

### Installation 

Prerequisites: Python 3.11 or higher
Package Mangaer: pipenv (in case of any others the instructions need to change accordingly)

Install the required libraries.

```
pipenv install openai scikit-learn pandas flask opensearch-py sentence-transformers lightgbm scikit-learn datasets transformers[torch] 
pipenv install --dev tqdm ipywidgets python-dotenv minsearch

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

#### Opensearch 

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


#### Search Options Evaluation

```zsh
pipenv run python src/run_comprehensive_evaluation.py

Loading .env environment variables...
COMPREHENSIVE EVALUATION
============================================================
Loading dataset from data/netflix_titles_cleaned.csv...
Sampled 1000 assets
Generated ground truth for 1000 assets
Total queries: 12000
Saved to: comprehensive_ground_truth.json
Created evaluation subset:
  Assets: 200
  Total queries: 2,400

Testing: MinSearch_Default
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 418.53it/s]
MinSearch_Default - HR@10: 0.6940, MRR@10: 0.5613, Time: 9.6ms

Testing: MinSearch_Optimized
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 386.61it/s]
MinSearch_Optimized - HR@10: 0.7280, MRR@10: 0.5777, Time: 10.4ms

Testing: Adaptive_Default
Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 409.32it/s]
Adaptive_Default - HR@10: 0.8620, MRR@10: 0.6160, Time: 9.8ms

Testing: OpenSearch_Hybrid

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:24<00:00, 20.35it/s]
OpenSearch_Hybrid - HR@10: 0.3960, MRR@10: 0.2914, Time: 196.2ms

================================================================================
COMPREHENSIVE EVALUATION RESULTS
================================================================================

Configuration        HR@1     HR@5     HR@10    MRR@10   Avg Time  
--------------------------------------------------------------------------------
MinSearch_Default    0.5100   0.6180   0.6940   0.5613   9.6       
MinSearch_Optimized  0.5140   0.6580   0.7280   0.5777   10.4      
Adaptive_Default     0.5140   0.7680   0.8620   0.6160   9.8       
OpenSearch_Hybrid    0.2120   0.3900   0.3960   0.2914   196.2     

BEST CONFIGURATION: Adaptive_Default (MRR@10: 0.6160)

================================================================================
DETAILED ANALYSIS: Adaptive_Default
================================================================================

Performance by Intent Type:
Intent          Count  HR@10    MRR@10  
----------------------------------------
genre_search    88     0.9205   0.8750  
title_search    249    0.7992   0.5215  
actor_search    106    0.9623   0.4965  
unknown         53     0.8868   0.8774  
director_search 2      1.0000   1.0000  
year_search     2      0.0000   0.0000  

Results saved to: comprehensive_evaluation_results.json
```
