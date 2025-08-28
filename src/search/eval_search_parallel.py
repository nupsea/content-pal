#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from typing import Dict, List, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm
from opensearchpy import OpenSearch

# Reuse your existing helpers
from index_assets import make_client
from hybrid_search import hybrid_search, rerank_topk   # rerank_topk is optional

# ----------------- metrics -----------------

def hit_rate_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0

def mrr_at_k(ranked_ids: List[str], gold_id: str, k: int = 10) -> float:
    for i, sid in enumerate(ranked_ids[:k], start=1):
        if sid == gold_id:
            return 1.0 / i
    return 0.0

# ----------------- workers -----------------

def _worker_retrieve_no_ce(
    client: OpenSearch,
    index: str,
    gold_id: str,
    query: str,
    pool_top_k: int,
    top_k: int,
) -> Tuple[float, float, int]:
    """
    Retrieve (BM25-PRF + E5 + RRF), compute HR/MRR for this pair, return partials.
    """
    out = hybrid_search(client, index, query, top_k=max(pool_top_k, top_k), return_prf=False)
    hits = out.get("hits", [])
    ids = [(h.get("_source", {}) or {}).get("show_id", "") or h.get("_id", "") for h in hits][:top_k]
    return hit_rate_at_k(ids, gold_id, k=top_k), mrr_at_k(ids, gold_id, k=top_k), 1

def _worker_retrieve_collect(
    client: OpenSearch,
    index: str,
    gold_id: str,
    query: str,
    pool_top_k: int,
) -> Tuple[str, str, List[dict]]:
    """
    Retrieve (BM25-PRF + E5 + RRF) and return raw hits for later cross-encoder rerank.
    """
    out = hybrid_search(client, index, query, top_k=pool_top_k, return_prf=False)
    return gold_id, query, out.get("hits", [])


# ----------------- evaluation (parallel) -----------------

def evaluate_pairs_parallel(
    client: OpenSearch,
    index: str,
    qid_to_queries: Dict[str, List[str]],
    *,
    top_k: int = 10,
    pool_top_k: int = 50,          # size of the fused pool before taking top_k
    workers: int = 12,
    max_pairs: int | None = None,  # sample this many (gold,query) pairs for speed; e.g., 500
    subsample_qids: int | None = None,  # or sample this many gold ids (keeps all their queries)
    seed: int = 42,
    use_cross_encoder: bool = False,
    ce_batch_k: int | None = None,  # leave None to use 'top_k' for rerank size
) -> Dict[str, float]:
    """
    Parallel evaluator.
    - If use_cross_encoder=False: compute metrics directly in worker (fastest).
    - If use_cross_encoder=True: two-stage pipeline (retrieve in parallel, then rerank sequentially with tqdm).
    """
    # Build the list of (gold, query) tasks
    items: List[Tuple[str, List[str]]] = list(qid_to_queries.items())
    rng = random.Random(seed)

    if subsample_qids is not None and subsample_qids < len(items):
        items = rng.sample(items, subsample_qids)

    pairs: List[Tuple[str, str]] = []
    for sid, qs in items:
        for q in qs:
            if q and isinstance(q, str):
                pairs.append((sid, q))

    if max_pairs is not None and max_pairs < len(pairs):
        pairs = rng.sample(pairs, max_pairs)

    total_pairs = len(pairs)
    if total_pairs == 0:
        return {"hit_rate": 0.0, "mrr": 0.0, "n": 0}

    # -------- Path A: no cross-encoder (fastest) --------
    if not use_cross_encoder:
        hr = mrr = n = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_worker_retrieve_no_ce, client, index, sid, q, pool_top_k, top_k)
                for (sid, q) in pairs
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="retrieval"):
                try:
                    hri, mrri, ni = fut.result()
                    hr += hri; mrr += mrri; n += ni
                except Exception as e:
                    # count as a skip; you can log e if you like
                    pass

        return {"hit_rate": hr / max(1, n), "mrr": mrr / max(1, n), "n": n}

    # -------- Path B: with cross-encoder rerank --------
    # Stage 1: parallel retrieve and collect
    collected: List[Tuple[str, str, List[dict]]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_worker_retrieve_collect, client, index, sid, q, pool_top_k)
            for (sid, q) in pairs
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="retrieval"):
            try:
                collected.append(fut.result())
            except Exception:
                # skip failures
                pass

    # Stage 2: rerank with cross-encoder (sequential but batched inside CE predict)
    # We’ll reuse your rerank_topk helper which already batches internally.
    rerank_size = ce_batch_k or top_k
    hr = mrr = n = 0

    for gold_id, query, hits in tqdm(collected, desc="rerank", total=len(collected)):
        if not hits:
            continue
        reranked = rerank_topk(query, hits, k=rerank_size)
        ids = [(h.get("_source", {}) or {}).get("show_id", "") or h.get("_id", "") for h in reranked][:top_k]
        hr  += hit_rate_at_k(ids, gold_id, k=top_k)
        mrr += mrr_at_k(ids, gold_id, k=top_k)
        n   += 1

    return {"hit_rate": hr / max(1, n), "mrr": mrr / max(1, n), "n": n}


# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="netflix_assets_v5")
    ap.add_argument("--pairs", required=True, help="Path to ground_truth.json")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--pool_top_k", type=int, default=120, help="size of fused pool before final top_k")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_pairs", type=int, default=None, help="sample N (gold,query) pairs")
    ap.add_argument("--subsample_qids", type=int, default=None, help="sample N gold ids (keep their queries)")
    ap.add_argument("--use_cross_encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.pairs, "r") as f:
        qid_to_queries = json.load(f)

    client = make_client()

    metrics = evaluate_pairs_parallel(
        client,
        args.index,
        qid_to_queries,
        top_k=args.top_k,
        pool_top_k=args.pool_top_k,
        workers=args.workers,
        max_pairs=args.max_pairs,
        subsample_qids=args.subsample_qids,
        seed=args.seed,
        use_cross_encoder=args.use_cross_encoder,
    )
    print(metrics)


if __name__ == "__main__":
    main()
