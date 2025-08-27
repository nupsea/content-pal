"""
Hybrid search functionality (optional enhanced search)
"""

import os
from typing import Dict, List, Tuple
from functools import lru_cache

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# Configuration - Optimized for hybrid performance
PRF_SEED = 50
PRF_FINAL = 100   # Reduced from 350 - still hybrid but faster
ANN_K = 100       # Reduced from 400 - still hybrid but faster  
RRF_KCONST = 60.0
W_BM = 1.0
W_ANN = 1.0

STOP_WORDS = {
    "the", "a", "an", "and", "of", "to", "in", "on", "for", "with", "about",
    "film", "movie", "show", "series", "new", "season", "tv", "watch"
}

# Keep original model to match existing index vector dimensions (1024)
_EMB = SentenceTransformer("BAAI/bge-large-en-v1.5")


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Get cross encoder for reranking"""
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=16384)
def _qvec_cached(q: str) -> Tuple[float, ...]:
    """Cached embedding computation"""
    v = _EMB.encode([q], normalize_embeddings=True)[0]
    return tuple(float(x) for x in (v.tolist() if hasattr(v, "tolist") else v))


def _qvec(q: str) -> List[float]:
    """Get query vector"""
    return list(_qvec_cached(q))


def _msm_for(q: str) -> str:
    """Get minimum should match based on query length"""
    n = len(q.split())
    if n <= 2:
        return "1"
    if n <= 5:
        return "2<-25%"
    return "3<-30%"


def bm25_prf(client: OpenSearch, index: str, query: str, 
             seed_size: int = PRF_SEED, final_size: int = PRF_FINAL) -> Dict:
    """BM25 search with simplified query expansion"""
    msm = _msm_for(query)
    
    # Simple multi-match search without complex PRF
    body = {
        "size": final_size,
        "query": {
            "multi_match": {
                "query": query,
                "type": "best_fields",
                "fields": ["title^8", "cast^5", "listed_in^4", "description^1.2"],
                "minimum_should_match": msm
            }
        }
    }
    
    return client.search(index=index, body=body)


def knn_candidates(client: OpenSearch, index: str, query: str, k: int = ANN_K) -> Dict:
    """Get KNN candidates using vector search"""
    try:
        vec = _qvec(query)
        if not vec:
            return {"hits": {"hits": []}}
        body = {"size": k, "query": {"knn": {"vector": {"vector": vec, "k": k}}}}
        return client.search(index=index, body=body)
    except Exception as e:
        print(f"Vector search failed: {e}, falling back to empty results")
        return {"hits": {"hits": []}}


def _doc_key(hit: dict) -> str:
    """Extract document key"""
    source = hit.get("_source") or {}
    return source.get("show_id") or source.get("id") or hit.get("_id", "")


def _collect_ranked(hits: List[dict]) -> Dict[str, Tuple[int, float, dict]]:
    """Collect ranked results"""
    out = {}
    for rank, hit in enumerate(hits or [], 1):
        key = _doc_key(hit)
        if not key:
            continue
        score = float(hit.get("_score", 0.0))
        if key not in out:
            out[key] = (rank, score, hit)
    return out


def rrf_fuse(bm25_hits: List[dict], ann_hits: List[dict], 
            k_const: float = RRF_KCONST, top_k: int = 50,
            w_bm: float = W_BM, w_ann: float = W_ANN) -> List[dict]:
    """Fuse results using Reciprocal Rank Fusion"""
    bm = _collect_ranked(bm25_hits)
    ann = _collect_ranked(ann_hits)
    ids = set(bm.keys()) | set(ann.keys())
    
    fused = []
    for key in ids:
        score = 0.0
        if key in bm:
            score += w_bm * (1.0 / (k_const + bm[key][0]))
        if key in ann:
            score += w_ann * (1.0 / (k_const + ann[key][0]))
        
        hit = bm[key][2] if key in bm else ann[key][2]
        fused.append((score, hit))
    
    fused.sort(key=lambda t: t[0], reverse=True)
    return [hit for _, hit in fused[:top_k]]


def rerank_topk(query: str, hits: List[dict], k: int = 20) -> List[dict]:
    """Rerank top results using cross encoder"""
    if not hits:
        return []
    
    ce = _get_cross_encoder()
    pairs = []
    keep = []
    for hit in hits:
        source = hit.get("_source", {}) or {}
        title = source.get("title", "")
        desc = source.get("description", "")
        text = f"{title} - {desc}"
        pairs.append((query, text))
        keep.append(hit)
    
    scores = ce.predict(pairs, batch_size=32)
    score_hit_pairs = list(zip([float(s) for s in scores], keep))
    score_hit_pairs.sort(key=lambda t: t[0], reverse=True)
    return [hit for _, hit in score_hit_pairs[:k]]


def hybrid_search(client: OpenSearch, index: str, query: str, top_k: int = 50) -> List[dict]:
    """Optimized hybrid search with parallel execution"""
    import concurrent.futures
    
    # Execute BM25 and vector search in parallel for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(bm25_prf, client, index, query)
        ann_future = executor.submit(knn_candidates, client, index, query)
        
        # Get results
        bm25_result = bm25_future.result()
        ann_result = ann_future.result()
    
    bm25_hits = bm25_result.get("hits", {}).get("hits", [])
    ann_hits = ann_result.get("hits", {}).get("hits", [])
    
    # Fuse results using RRF - maintains hybrid nature
    fused_hits = rrf_fuse(bm25_hits, ann_hits, top_k=top_k)
    
    return fused_hits