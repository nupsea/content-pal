from __future__ import annotations
import json, random
from typing import Dict, List, Tuple
from opensearchpy import OpenSearch

# import your existing helpers
from index_assets import make_client
from hybrid_search import hybrid_search, rerank_topk

# ---- metrics ----

def hit_rate_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    """1.0 if gold_id appears in top-k, else 0.0"""
    return 1.0 if gold_id in ranked_ids[:k] else 0.0

def mrr_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    """Reciprocal rank of gold_id within top-k (0.0 if missing)."""
    for i, sid in enumerate(ranked_ids[:k], start=1):
        if sid == gold_id:
            return 1.0 / i
    return 0.0

# ---- evaluation harness ----

def evaluate_pairs(client, index: str, qid_to_queries: dict[str, list[str]],
                   *, top_k: int = 10, use_cross_encoder: bool = False,
                   subsample: int | None = None, seed: int = 42) -> dict[str, float]:
    import random
    items = list(qid_to_queries.items())
    if subsample and subsample < len(items):
        random.seed(seed)
        items = random.sample(items, subsample)

    hr = mrr = n = 0
    for gold_id, queries in items:
        for q in queries:
            out = hybrid_search(client, index, q, top_k=max(50, top_k), return_prf=False)
            hits = out.get("hits", [])
            hits = rerank_topk(q, hits, k=top_k) if use_cross_encoder else hits[:top_k]
            ids = [h.get("_source", {}).get("show_id", "") or h.get("_id", "") for h in hits]
            hr  += hit_rate_at_k(ids, gold_id, k=top_k)
            mrr += mrr_at_k(ids, gold_id, k=top_k)
            n   += 1

    return {"hit_rate": hr / max(1, n), "mrr": mrr / max(1, n), "n": n}


# ---- CLI usage ----

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="netflix_assets_v4")
    ap.add_argument("--pairs", required=True, help="path to ground_truth.json")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--subsample", type=int, default=None, help="evaluate on N items (for speed)")
    ap.add_argument("--use_cross_encoder", action="store_true")
    args = ap.parse_args()

    with open(args.pairs, "r") as f:
        qid_to_queries = json.load(f)

    client = make_client()
    metrics = evaluate_pairs(
        client, args.index, qid_to_queries,
        top_k=args.top_k,
        use_cross_encoder=args.use_cross_encoder,
        subsample=args.subsample,
    )
    print(metrics)
