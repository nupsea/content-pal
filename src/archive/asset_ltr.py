from __future__ import annotations
import re, math, json
from functools import lru_cache
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
RET = {
    "embedder": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # 384-dim, good for short phrases
    "bm25_size": 300,   # candidate pool from BM25
    "top_k": 10,        # final results
    "expand": True      # tiny generic expansions (no LLM)
}

# ---------- Tiny normalization/expansion ----------
def _normalize_expand(q: str, enable=True) -> str:
    qn = q.lower().strip().replace("sci-fi","sci fi").replace("&"," and ")
    if not enable: return qn
    adds = []
    if "kids" in qn or "family" in qn: adds += ["children","preschool","toddler","family"]
    if any(t in qn for t in ("romantic","romance","romcom")): adds += ["love","relationships","romcom"]
    if any(t in qn for t in ("mature","adult")): adds += ["tv ma","r","mature"]
    if any(t in qn for t in ("series"," tv "," tv-"," tv_"," show")): adds += ["tv show","series"]
    for hint,country in [("german","germany"),("korean","south korea"),("japanese","japan"),
                         ("french","france"),("spanish","spain"),("italian","italy")]:
        if hint in qn: adds.append(country)
    return qn + (" " + " ".join(adds) if adds else "")

# ---------- Embedder ----------
@lru_cache(maxsize=1)
def get_embedder(name: str):
    return SentenceTransformer(name)

# ---------- BM25 candidates (dis_max “intent buckets”) ----------
# def bm25_query(qn: str) -> Dict:
#     return {
#       "dis_max": {
#         "tie_breaker": 0.10,
#         "queries": [
#           {"multi_match": {"query": qn, "type": "phrase",
#                            "fields": ["title^12","cast^8"], "slop": 1}},
#           {"multi_match": {"query": qn, "type": "best_fields",
#                            "fields": ["title^10","cast^7","director^2.5"],
#                            "operator": "AND", "minimum_should_match": "2<-25%"}},
#           {"multi_match": {"query": qn, "type": "best_fields",
#                            "fields": ["listed_in_text^9"], "operator": "OR"}},
#           {"multi_match": {"query": qn, "type": "best_fields",
#                            "fields": ["country_text^4","rating_text^2.5","type_text^1.5"]}},
#           {"match_bool_prefix": {"title": {"query": qn, "boost": 2.0}}},
#           {"match_bool_prefix": {"cast":  {"query": qn, "boost": 1.5}}},
#           {"match_phrase": {"description": {"query": qn, "slop": 2, "boost": 1.05}}}
#         ]
#       }
#     }

def bm25_query(qn: str) -> dict:
    return {
      "dis_max": {
        "tie_breaker": 0.10,
        "queries": [
          # Names (title/cast): phrase + most_fields with prefix subfields
          {"multi_match": {
              "query": qn, "type": "phrase",
              "fields": ["title^12","cast^8"], "slop": 1
          }},
          {"multi_match": {
              "query": qn, "type": "most_fields",
              "fields": ["title^10","title.prefix^6","cast^7","cast.prefix^4","director^2.5"],
              "minimum_should_match": "2<-30%"
          }},

          # Consolidated blob (english + shingles) for paraphrases / multiword cues
          {"match": {"search_blob": {"query": qn, "operator": "OR", "boost": 6.0}}},
          {"match_phrase": {"search_blob": {"query": qn, "slop": 1, "boost": 8.0}}},

          # Categories / audience / country / rating / type with synonyms
          {"multi_match": {
              "query": qn, "type": "cross_fields",
              "fields": ["listed_in_text^9","country_text^4","rating_text^2.5","type_text^2"]
          }},

          # Extra phrase channel for description shingles (helps "book club", "cute animals", etc.)
          {"multi_match": {
              "query": qn, "type": "best_fields",
              "fields": ["description.shingles^6"]
          }},

          # Prefix-friendly quick wins
          {"match_bool_prefix": {"title": {"query": qn, "boost": 2.0}}},
          {"match_bool_prefix": {"cast":  {"query": qn, "boost": 1.5}}}
        ]
      }
    }


def bm25_candidates(client, index: str, q: str, *, cfg: Dict = RET):
    qn = _normalize_expand(q, cfg["expand"])
    body = {
        "size": cfg["bm25_size"],
        "_source": [
            "show_id","title","type","release_year","listed_in","listed_in_text",
            "cast","director","country","country_text","rating","rating_text","description","vector"
        ],
        "query": bm25_query(qn)
    }
    return client.search(index=index, body=body)

# ---------- Features ----------
_tok = re.compile(r"[a-z0-9']+")
def toks(s: str) -> set:
    return set(_tok.findall((s or "").lower()))

def char3(s: str) -> set:
    s = (s or "").lower()
    return set(s[i:i+3] for i in range(len(s)-2)) if len(s) >= 3 else set()

def overlap(a: set, b: set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a)

def features_for(query: str, doc: Dict, bm25_score: float, qvec=None) -> List[float]:
    qt = toks(query); q3 = char3(query)

    title = doc.get("title","")
    cast  = doc.get("cast","")
    listed= ", ".join(doc.get("listed_in") or []) or doc.get("listed_in_text","") or ""
    desc  = doc.get("description","")
    country = doc.get("country_text", doc.get("country","") or "")
    rating  = doc.get("rating_text",  doc.get("rating","")  or "")
    dtype   = doc.get("type","")

    ft = toks(title); fc = toks(cast)
    fg = toks(listed); fd = toks(desc)
    fcountry = toks(country); frating = toks(rating)

    ov_title = overlap(qt, ft)
    ov_cast  = overlap(qt, fc)
    ov_genre = overlap(qt, fg)
    ov_desc  = overlap(qt, fd)

    p_title = overlap(q3, char3(title))
    p_cast  = overlap(q3, char3(cast))

    want_movie = 1.0 if any(w in qt for w in ("movie","film")) else 0.0
    want_series= 1.0 if any(w in qt for w in ("series","show","tv")) else 0.0
    is_movie   = 1.0 if dtype.lower().startswith("movie") else 0.0
    is_series  = 1.0 if "tv" in dtype.lower() else 0.0

    has_country = 1.0 if (qt & fcountry) else 0.0
    mature_q    = 1.0 if any(w in qt for w in ("mature","adult","tv","ma","r","nc-17","nc17")) else 0.0
    is_mature   = 1.0 if any(x in (rating or "").lower() for x in ("tv-ma","r","nc-17")) else 0.0

    cos = 0.0
    v = doc.get("vector")
    if qvec is not None and isinstance(v, list) and v:
        # stored vectors should be L2-normalized -> dot == cosine
        cos = float(sum(a*b for a,b in zip(qvec, v)))

    q_len = float(len(qt))
    title_len = float(len(ft))

    return [
        float(bm25_score), cos,
        ov_title, ov_cast, ov_genre, ov_desc,
        p_title, p_cast,
        want_movie, want_series, is_movie, is_series,
        has_country, mature_q, is_mature,
        q_len, title_len
    ]

FEATURE_NAMES = [
    "bm25","cosine","ov_title","ov_cast","ov_genre","ov_desc",
    "p_title","p_cast","want_movie","want_series","is_movie","is_series",
    "has_country","mature_q","is_mature","q_len","title_len"
]

# ---------- Training data builder ----------
def build_training_matrix(client, index: str, qid_to_queries: Dict[str, List[str]], *, cfg: Dict = RET):
    emb = get_embedder(cfg["embedder"])
    X, y, group = [], [], []
    # each query = one group
    for gold_id, queries in qid_to_queries.items():
        for q in queries:
            res = bm25_candidates(client, index, q, cfg=cfg)
            hits = res.get("hits",{}).get("hits",[])
            if not hits: 
                continue
            qvec = emb.encode(q, normalize_embeddings=True).tolist()

            rowX, rowY = [], []
            # keep only if pool contains the positive — improves training signal
            contains_pos = False
            for h in hits:
                src = h.get("_source", {})
                bm = float(h.get("_score", 0.0))
                rowX.append(features_for(q, src, bm, qvec))
                is_pos = 1.0 if src.get("show_id") == gold_id else 0.0
                if is_pos == 1.0: contains_pos = True
                rowY.append(is_pos)
            if not contains_pos:
                continue
            X.extend(rowX); y.extend(rowY); group.append(len(rowX))
    return np.array(X, np.float32), np.array(y, np.float32), np.array(group, np.int32)

# ---------- Train / save / load ----------
def train_ltr(client, index: str, qid_to_queries: Dict[str,List[str]], *, cfg: Dict = RET,
              num_leaves=63, lr=0.05, n_estimators=400):
    import lightgbm as lgb
    X, y, group = build_training_matrix(client, index, qid_to_queries, cfg=cfg)
    train_set = lgb.Dataset(X, label=y, group=group, feature_name=FEATURE_NAMES)
    params = dict(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
        learning_rate=lr, num_leaves=num_leaves, min_data_in_leaf=20
    )
    model = lgb.train(params, train_set, num_boost_round=n_estimators)
    return model

def save_model(model, path: str) -> None:
    model.save_model(path)

def load_model(path: str):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)

# ---------- Inference ----------
def retrieve_with_ltr(client, index: str, q: str, model, *, cfg: Dict = RET):
    res = bm25_candidates(client, index, q, cfg=cfg)
    hits = res.get("hits",{}).get("hits",[])
    if not hits:
        return []
    emb = get_embedder(cfg["embedder"])
    qvec = emb.encode(q, normalize_embeddings=True).tolist()

    X = [features_for(q, h.get("_source",{}), float(h.get("_score",0.0)), qvec) for h in hits]
    X = np.array(X, np.float32)
    scores = model.predict(X)
    ranked = sorted(zip(scores, hits), key=lambda t: t[0], reverse=True)
    return [h for _, h in ranked[:cfg["top_k"]]]

# ---------- Metrics ----------
def ids_from_hits(hits: List[Dict]) -> List[str]:
    return [h["_source"].get("show_id","") for h in hits]

def hit_rate_at_k(ids: List[str], gold: str, k: int=10) -> float:
    return 1.0 if gold in ids[:k] else 0.0

def mrr_at_k(ids: List[str], gold: str, k: int=10) -> float:
    for i,x in enumerate(ids[:k], 1):
        if x == gold: return 1.0 / i
    return 0.0
