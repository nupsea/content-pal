
import os
from functools import lru_cache
from typing import Dict, List, Tuple
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# ---- small, fast model ----
_EMB = SentenceTransformer(os.getenv("HYB_EMB", "BAAI/bge-large-en-v1.5"))

# ---- ENV knobs (read once) ----
PRF_SEED      = int(os.getenv("HYB_BM25_SEED", "80"))
PRF_FINAL     = int(os.getenv("HYB_BM25_FINAL", "350"))
PRF_EXP_BOOST = float(os.getenv("HYB_EXP_BOOST", "0.6"))
PRF_SIG_MIN   = float(os.getenv("HYB_PRF_SIG", "1.2"))
PRF_MAX_TERMS = int(os.getenv("HYB_PRF_MAX_TERMS", "12"))

ANN_K         = int(os.getenv("HYB_ANN_K", "400"))
RRF_KCONST    = float(os.getenv("HYB_RRF_K", "60"))
W_BM          = float(os.getenv("HYB_W_BM", "1.0"))
W_ANN         = float(os.getenv("HYB_W_ANN", "1.0"))

STOP = {
    "the","a","an","and","of","to","in","on","for","with","about",
    "film","movie","show","series","new","season","tv","watch"
}

_CE_PATH = os.getenv("CE_PATH", "ce_netflix")  # folder from training

@lru_cache(maxsize=1)
def _get_ce():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(_CE_PATH)
    except Exception as e:
        print("[rerank] WARN: falling back to base CE:", e)
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_topk(query: str, hits: List[dict], k: int = 20) -> List[dict]:
    if not hits:
        return []
    ce = _get_ce()
    pairs = []
    keep: List[dict] = []
    for h in hits:
        s = h.get("_source", {}) or {}
        title = s.get("title","")
        desc  = s.get("description","")
        text  = f"{title} — {desc}"
        pairs.append((query, text))
        keep.append(h)
    scores = ce.predict(pairs, batch_size=32)
    # Pylance-friendly sort
    score_hit_pairs: List[Tuple[float, dict]] = list(zip([float(x) for x in scores], keep))
    score_hit_pairs.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in score_hit_pairs[:k]]


# ---- cached embedding ----
@lru_cache(maxsize=16384)
def _qvec_cached(q: str) -> Tuple[float, ...]:
    v = _EMB.encode([q], normalize_embeddings=True)[0]  # BGE-large doesn't need query prefix
    return tuple(float(x) for x in (v.tolist() if hasattr(v, "tolist") else v))

def _qvec(q: str) -> List[float]:
    return list(_qvec_cached(q))

def _msm_for(q: str) -> str:
    n = len(q.split())
    if n <= 2:  return "1"
    if n <= 5:  return "2<-25%"
    return "3<-30%"

def bm25_prf(client: OpenSearch, index: str, q: str,
             *, seed_size: int = PRF_SEED, final_size: int = PRF_FINAL,
             expand_boost: float = PRF_EXP_BOOST, sig_min: float = PRF_SIG_MIN,
             max_terms: int = PRF_MAX_TERMS) -> Dict:
    msm = _msm_for(q)

    seed_body = {
      "size": seed_size,
      "query": {
        "multi_match": {
          "query": q,
          "type": "best_fields",
          # include expanded_text (low boost in seed)
          "fields": ["title^6","cast^4","listed_in_text^3","description","expanded_text^2.5"],
          "minimum_should_match": msm
        }
      },
      "aggs": {
        "sig_title":  {"significant_text": {"field": "title",         "size": 6, "min_doc_count": 2}},
        "sig_desc":   {"significant_text": {"field": "description",   "size": 8, "min_doc_count": 2}},
        "sig_listed": {"significant_text": {"field": "listed_in_text","size": 6, "min_doc_count": 2}},
        "sig_cast":   {"significant_text": {"field": "cast",          "size": 6, "min_doc_count": 2}},
        # we generally do NOT mine from expanded_text (it’s synthetic), keep it clean
      }
    }
    seed = client.search(index=index, body=seed_body)

    terms: list[str] = []
    for agg in ("sig_title","sig_desc","sig_listed","sig_cast"):
        for b in seed.get("aggregations", {}).get(agg, {}).get("buckets", []):
            t = (b.get("key","") or "").strip().lower()
            sc = float(b.get("score", 0.0))
            if len(t) >= 3 and t not in STOP and " " not in t and sc >= sig_min:
                terms.append(t)
    # dedupe, keep order
    seen = set(); terms = [t for t in terms if not (t in seen or seen.add(t))]
    terms = terms[:max_terms]

    expanded_q = q + " " + " ".join(terms) if terms else q

    # --- Final BM25: base + expanded branch + phrase + prefix
    body = {
      "size": final_size,
      "query": {
        "dis_max": {
          "tie_breaker": 0.1,
          "queries": [
            # base
            {"multi_match": {
              "query": q, "type": "best_fields",
              "fields": ["title^8","cast^5","listed_in_text^4","description^1.2","expanded_text^3"],
              "minimum_should_match": msm
            }},
            # PRF-expanded
            {"multi_match": {
              "query": expanded_q, "type": "best_fields",
              "fields": ["title^4","cast^3","listed_in_text^3","description","expanded_text^2.5"],
              "boost": expand_boost,
              "minimum_should_match": msm
            }},
            # phrase (helps short exact-ish queries)
            {"multi_match": {
              "query": q, "type": "phrase", "slop": 1,
              "fields": ["title^10","cast^6"],
              "boost": 1.2
            }},
            # prefix on title (one/two-word queries)
            {"multi_match": {
              "query": q, "type": "phrase_prefix", "slop": 1, "max_expansions": 20,
              "fields": ["title^4"]
            }}
          ]
        }
      }
    }

    res = client.search(index=index, body=body)
    res["_prf_terms"] = terms
    return res


def knn_candidates(client: OpenSearch, index: str, q: str, *, k: int = ANN_K) -> Dict:
    vec = _qvec(q)
    body = {"size": k, "query": {"knn": {"vector": {"vector": vec, "k": k}}}}
    return client.search(index=index, body=body)

# ---- robust & weighted RRF ----
def _doc_key(hit: dict) -> str | None:
    src = hit.get("_source") or {}
    sid = src.get("show_id")
    return sid or hit.get("_id")

def _collect_ranked(hits: List[dict]) -> Dict[str, Tuple[int, float, dict]]:
    out: Dict[str, Tuple[int, float, dict]] = {}
    for rank, h in enumerate(hits or [], 1):
        key = _doc_key(h)
        if not key:
            continue
        try:
            sc = float(h.get("_score", 0.0))
        except Exception:
            sc = 0.0
        if key not in out:
            out[key] = (rank, sc, h)
    return out

def rrf_fuse(bm25_hits: List[dict], ann_hits: List[dict], *,
             k_const: float = RRF_KCONST, top_k: int = 50,
             w_bm: float = W_BM, w_ann: float = W_ANN) -> List[dict]:
    bm = _collect_ranked(bm25_hits)
    ann = _collect_ranked(ann_hits)
    ids = set(bm.keys()) | set(ann.keys())
    fused: List[Tuple[float, dict]] = []
    for key in ids:
        s = 0.0
        if key in bm:  s += w_bm  * (1.0 / (k_const + bm[key][0]))
        if key in ann: s += w_ann * (1.0 / (k_const + ann[key][0]))
        h = bm[key][2] if key in bm else ann[key][2]
        fused.append((s, h))
    fused.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in fused[:top_k]]

def hybrid_search(client: OpenSearch, index: str, q: str, *,
                  top_k: int = 50,
                  bm25_seed: int = PRF_SEED,
                  bm25_final: int = PRF_FINAL,
                  ann_k: int = ANN_K,
                  return_prf: bool = False) -> Dict:
    # allow overrides via args or env
    bm = bm25_prf(client, index, q, seed_size=bm25_seed, final_size=bm25_final,
                  expand_boost=PRF_EXP_BOOST, sig_min=PRF_SIG_MIN, max_terms=PRF_MAX_TERMS)
    ann = knn_candidates(client, index, q, k=ann_k)
    fused = rrf_fuse(bm.get("hits", {}).get("hits", []),
                     ann.get("hits", {}).get("hits", []),
                     k_const=RRF_KCONST, top_k=top_k,
                     w_bm=W_BM, w_ann=W_ANN)
    out = {"hits": fused}
    if return_prf:
        out["prf_terms"] = bm.get("_prf_terms", [])
    return out

# -------- Optional cross-encoder re-rank (extra MRR bump) --------
# try:
#     from sentence_transformers import CrossEncoder
#     _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# except Exception:
#     _CE = None

# def rerank_topk(query: str, hits: list[dict], k: int = 20) -> list[dict]:
#     if not hits or _CE is None:
#         return hits[:k]
#     pairs = []
#     for h in hits:
#         s = h.get("_source", {})
#         text = (s.get("title", "") or "") + " — " + (s.get("description", "") or "")
#         pairs.append((query, text))
#     scores = _CE.predict(pairs, batch_size=32)  # numpy array
#     score_hit_pairs: list[tuple[float, dict]] = list(zip([float(s) for s in scores], hits))
#     score_hit_pairs.sort(key=lambda t: t[0], reverse=True)
#     return [h for _, h in score_hit_pairs[:k]]
