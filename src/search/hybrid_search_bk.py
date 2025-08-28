from __future__ import annotations
from typing import Dict, List, Tuple
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer



# small model; fast and solid
_EMB = SentenceTransformer("intfloat/e5-small-v2")

def _qvec(q: str) -> List[float]:
    return _EMB.encode(["query: " + q], normalize_embeddings=True)[0].tolist()

STOP = {
    "the","a","an","and","of","to","in","on","for","with","about",
    "film","movie","show","series","new","season","tv","watch"
}

def bm25_prf(client, index: str, q: str, *, seed_size=80, final_size=300, expand_boost=0.6) -> Dict:
    seed_body = {
      "size": seed_size,
      "query": {
        "multi_match": {
          "query": q,
          "type": "best_fields",
          "fields": ["title^6","cast^4","listed_in_text^3","description"],
          "minimum_should_match": "2<-25%"
        }
      },
      "aggs": {
        "sig_title":  {"significant_text": {"field": "title",         "size": 6, "min_doc_count": 2}},
        "sig_desc":   {"significant_text": {"field": "description",   "size": 8, "min_doc_count": 2}},
        "sig_listed": {"significant_text": {"field": "listed_in_text","size": 6, "min_doc_count": 2}},
        "sig_cast":   {"significant_text": {"field": "cast",          "size": 6, "min_doc_count": 2}}
      }
    }
    seed = client.search(index=index, body=seed_body)

    # collect better terms
    terms: list[str] = []
    for agg in ("sig_title","sig_desc","sig_listed","sig_cast"):
        for b in seed.get("aggregations", {}).get(agg, {}).get("buckets", []):
            t = (b.get("key","") or "").strip().lower()
            sc = float(b.get("score", 0.0))
            if len(t) >= 3 and t not in STOP and " " not in t and sc >= 1.2:
                terms.append(t)
    # dedupe, keep order
    seen = set(); terms = [t for t in terms if not (t in seen or seen.add(t))]
    terms = terms[:12]

    expanded_q = q + " " + " ".join(terms) if terms else q
    body = {
      "size": final_size,
      "query": {
        "dis_max": {
          "tie_breaker": 0.1,
          "queries": [
            {"multi_match": {
              "query": q, "type": "best_fields",
              "fields": ["title^8","cast^5","listed_in_text^4","description^1.2"],
              "minimum_should_match": "2<-25%"
            }},
            {"multi_match": {
              "query": expanded_q, "type": "best_fields",
              "fields": ["title^4","cast^3","listed_in_text^3","description"],
              "boost": expand_boost,
              "minimum_should_match": "2<-25%"
            }}
          ]
        }
      }
    }
    res = client.search(index=index, body=body, _source=True)
    res["_prf_terms"] = terms
    return res

def knn_candidates(client: OpenSearch, index: str, q: str, *, k=300) -> Dict:
    vec = _qvec(q)
    body = {"size": k, "query": {"knn": {"vector": {"vector": vec, "k": k}}}}
    return client.search(index=index, body=body)

def _doc_key(hit: dict) -> str | None:
    """Prefer catalog show_id; fall back to ES _id."""
    src = hit.get("_source") or {}
    sid = src.get("show_id")
    return sid or hit.get("_id")

def _collect_ranked(hits: list[dict]) -> dict[str, tuple[int, float, dict]]:
    out: dict[str, tuple[int, float, dict]] = {}
    for rank, h in enumerate(hits, 1):
        key = _doc_key(h)
        if not key:
            continue
        # score can be absent in rare cases; treat as 0.0
        try:
            sc = float(h.get("_score", 0.0))
        except Exception:
            sc = 0.0
        # keep the first (best-ranked) occurrence
        if key not in out:
            out[key] = (rank, sc, h)
    return out

def rrf_fuse(bm25_hits: list[dict], ann_hits: list[dict], *, k_const: float = 60.0, top_k: int = 50) -> list[dict]:
    bm = _collect_ranked(bm25_hits or [])
    ann = _collect_ranked(ann_hits or [])

    ids = set(bm.keys()) | set(ann.keys())
    fused: list[tuple[float, dict]] = []

    for key in ids:
        s = 0.0
        if key in bm:
            s += 1.0 / (k_const + bm[key][0])
        if key in ann:
            s += 1.0 / (k_const + ann[key][0])
        # choose the representative hit from whichever channel has it
        h = bm[key][2] if key in bm else ann[key][2]
        fused.append((s, h))

    fused.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in fused[:top_k]]


def hybrid_search(client: OpenSearch, index: str, q: str, *,
                  top_k=50, bm25_seed=80, bm25_final=350, ann_k=400, return_prf=False) -> Dict:
    bm  = bm25_prf(client, index, q, seed_size=bm25_seed, final_size=bm25_final)
    ann = knn_candidates(client, index, q, k=ann_k)
    fused = rrf_fuse(bm.get("hits", {}).get("hits", []), ann.get("hits", {}).get("hits", []),
                     k_const=60, top_k=top_k)
    out = {"hits": fused}
    if return_prf:
        out["prf_terms"] = bm.get("_prf_terms", [])
    return out

# -------- Optional cross-encoder re-rank (extra MRR bump) --------
try:
    from sentence_transformers import CrossEncoder
    _CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _CE = None

def rerank_topk(query: str, hits: list[dict], k: int = 20) -> list[dict]:
    if not hits or _CE is None:
        return hits[:k]
    pairs = []
    for h in hits:
        s = h.get("_source", {})
        text = (s.get("title", "") or "") + " — " + (s.get("description", "") or "")
        pairs.append((query, text))
    scores = _CE.predict(pairs, batch_size=32)  # numpy array
    score_hit_pairs: list[tuple[float, dict]] = list(zip([float(s) for s in scores], hits))
    score_hit_pairs.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in score_hit_pairs[:k]]

