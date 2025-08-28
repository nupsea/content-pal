#!/usr/bin/env python3
from __future__ import annotations
import json, random, os
from typing import Dict, List, Tuple
from opensearchpy import OpenSearch
from tqdm.auto import tqdm

# reuse your client + hybrid bits
from index_assets import make_client
from hybrid_search import bm25_prf, knn_candidates

INDEX = os.getenv("OS_INDEX", "netflix_assets_v5")
PAIRS = os.getenv("GT_PATH",  "../../notebooks/ground_truth.json")
OUT   = os.getenv("OUT_PATH", "ce_train_pairs.jsonl")

# how many negatives & pools to mine
BM25_SIZE = int(os.getenv("BM25_SIZE", "120"))
ANN_K     = int(os.getenv("ANN_K", "200"))
NEG_PER_POS = int(os.getenv("NEG_PER_POS", "4"))
SEED = int(os.getenv("SEED", "42"))

def collect_text(hit: dict) -> Tuple[str, str]:
    """Return (id, text) for CE (title — description)."""
    src = hit.get("_source", {}) or {}
    sid = src.get("show_id") or hit.get("_id", "")
    title = src.get("title", "") or ""
    desc  = src.get("description", "") or ""
    text  = f"{title} — {desc}"
    return sid, text

def main():
    client: OpenSearch = make_client()
    pairs: Dict[str, List[str]] = json.load(open(PAIRS, "r"))
    rng = random.Random(SEED)

    out = open(OUT, "w")
    total, built = 0, 0

    for gold_id, queries in tqdm(pairs.items(), desc="build"):
        for q in queries:
            if not q or not isinstance(q, str): 
                continue

            # candidate pools (lexical + vector)
            bm = bm25_prf(client, INDEX, q, seed_size=80, final_size=BM25_SIZE).get("hits", {}).get("hits", [])
            ann = knn_candidates(client, INDEX, q, k=ANN_K).get("hits", {}).get("hits", [])

            pool = {}
            for h in bm + ann:
                sid, text = collect_text(h)
                if sid and text:
                    pool[sid] = text

            # positive
            pos_text = pool.get(gold_id)
            if not pos_text:
                continue

            # negatives: choose from pool without gold
            neg_ids = [sid for sid in pool.keys() if sid != gold_id]
            rng.shuffle(neg_ids)
            neg_ids = neg_ids[:NEG_PER_POS]
            if not neg_ids:
                continue

            # write pointwise pairs (label 1 for pos, 0 for negs)
            # (You can switch to pairwise later; pointwise is super simple)
            total += 1
            ex_pos = {"q": q, "d": pos_text, "label": 1.0}
            out.write(json.dumps(ex_pos) + "\n")
            built += 1

            for nid in neg_ids:
                ex_neg = {"q": q, "d": pool[nid], "label": 0.0}
                out.write(json.dumps(ex_neg) + "\n")

    out.close()
    print(f"wrote: {OUT}  examples≈{built*(1+NEG_PER_POS)}  queries={built}")

if __name__ == "__main__":
    main()
