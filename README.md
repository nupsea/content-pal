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



### Opensearch engine

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


### Run the below scripts

#### Index
```zsh
❯ python index_assets.py


modules.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 387/387 [00:00<00:00, 1.25MB/s]
README.md: 67.8kB [00:00, 232MB/s]
sentence_bert_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 57.0/57.0 [00:00<00:00, 193kB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 615/615 [00:00<00:00, 2.51MB/s]
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 133M/133M [00:11<00:00, 11.2MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 314/314 [00:00<00:00, 2.27MB/s]
vocab.txt: 232kB [00:00, 1.18MB/s]
tokenizer.json: 711kB [00:00, 35.9MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [00:00<00:00, 533kB/s]
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 630kB/s]
Created index netflix_assets_v4
Bulk indexed OK=7370, Fail=0
```


#### Evaluate in parallel
```zsh
cd /src/search

❯ python eval_search_parallel.py \
  --index netflix_assets_v4 \
  --pairs ../../notebooks/ground_truth.json \
  --top_k 10 \
  --pool_top_k 80 \
  --workers 12 \
  --max_pairs 1000 \
  --use_cross_encoder

retrieval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:38<00:00, 26.12it/s]
rerank: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:05<00:00, 15.28it/s]
{'hit_rate': 0.37, 'mrr': 0.26932896825396824, 'n': 1000}
```

#### Sweep and find the best
```zsh
cd /src/search

❯ python sweep_hybrid.py


{'hit_rate': 0.372, 'mrr': 0.22022619047619052, 'n': 500, 'HYB_BM25_FINAL': 300, 'HYB_ANN_K': 300, 'HYB_EXP_BOOST': 0.5, 'HYB_RRF_K': 40.0, 'HYB_W_BM': 1.0, 'HYB_W_ANN': 1.0}
{'hit_rate': 0.37, 'mrr': 0.21588730158730157, 'n': 500, 'HYB_BM25_FINAL': 300, 'HYB_ANN_K': 300, 'HYB_EXP_BOOST': 0.5, 'HYB_RRF_K': 40.0, 'HYB_W_BM': 1.0, 'HYB_W_ANN': 1.2}
{'hit_rate': 0.372, 'mrr': 0.21982857142857143, 'n': 500, 'HYB_BM25_FINAL': 300, 'HYB_ANN_K': 300, 'HYB_EXP_BOOST': 0.5, 'HYB_RRF_K': 40.0, 'HYB_W_BM': 1.2, 'HYB_W_ANN': 1.0}
{'hit_rate': 0.372, 'mrr': 0.21995952380952383, 'n': 500, 'HYB_BM25_FINAL': 300, 'HYB_ANN_K': 300, 'HYB_EXP_BOOST': 0.5, 'HYB_RRF_K': 40.0, 'HYB_W_BM': 1.2, 'HYB_W_ANN': 1.2}
{'hit_rate': 0.372, 'mrr': 0.21945952380952383, 'n': 500, 'HYB_BM25_FINAL': 300, 'HYB_ANN_K': 300, 'HYB_EXP_BOOST': 0.5, 'HYB_RRF_K': 60.0, 'HYB_W_BM': 1.0, 'HYB_W_ANN': 1.0}
{'hit_rate': 0.364, 'mrr': 0.2152873015873016, 'n': 500, 'HYB_BM25_FINAL': 300, 'HYB_ANN_K': 300, 'HYB_EXP_BOOST': 0.5, 'HYB_RRF_K': 60.0, 'HYB_W_BM': 1.0, 'HYB_W_ANN': 1.2}
...

```

#### Generate expanded text (run once)

❯ python expand_doc2query.py
```zsh
tokenizer_config.json: 2.12kB [00:00, 2.75MB/s]
spiece.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 792k/792k [00:03<00:00, 258kB/s]
tokenizer.json: 1.39MB [00:00, 18.1MB/s]
special_tokens_map.json: 1.79kB [00:00, 3.57MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 665/665 [00:00<00:00, 1.83MB/s]
pytorch_model.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 308M/308M [00:31<00:00, 9.89MB/s]
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 308M/308M [00:30<00:00, 10.1MB/s]
doc2query: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 922/922 [07:52<00:00,  1.95it/s]
wrote ../../data/netflix_titles_expanded.csv   (rows=7370)
❯ python index_assets.py
```

#### Index and Eval Search in parallel

```zsh
❯ python index_assets.py
Created index netflix_assets_v5
Bulk indexed OK=7370, Fail=0
❯ python eval_search_parallel.py \
  --index netflix_assets_v5 \
  --pairs ../../notebooks/ground_truth.json

retrieval:  37%|███████████████████████████████████████████████████████████▌                                                                                                     | 1851/5000 [01:16<02:00, 26.05it/s]
```
