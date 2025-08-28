import json, argparse
from statistics import mean
from open_search import Cfg, make_client
from asset_ltr import (
    RET, bm25_candidates, ids_from_hits, train_ltr, save_model, load_model,
    retrieve_with_ltr, hit_rate_at_k, mrr_at_k
)

DEFAULT_INDEX = "netflix_assets_v2"
MODEL_PATH = "asset_ltr.bin"

def pool_recall(os_client, index, pairs, *, bm25_size, expand=True):
    """Recall@bm25_size: fraction of (q,gold) where BM25 pool contains gold."""
    tmp_cfg = {**RET, "bm25_size": bm25_size, "expand": expand}
    total, ok = 0, 0
    misses = []
    for gold, queries in pairs.items():
        for q in queries:
            total += 1
            res = bm25_candidates(os_client, index, q, cfg=tmp_cfg)
            ids = ids_from_hits(res.get("hits", {}).get("hits", []))
            if gold in ids:
                ok += 1
            else:
                if len(misses) < 20:
                    misses.append((gold, q, ids[:5]))
    recall = ok / max(1, total)
    return recall, misses

def auto_bm25_size(os_client, index, pairs, *, targets=(0.85, 0.9), sizes=(250, 300, 400, 500, 750, 1000)):
    """Try increasing BM25 pool size until pool recall hits target."""
    chosen = sizes[0]; last = 0.0
    for s in sizes:
        r, _ = pool_recall(os_client, index, pairs, bm25_size=s)
        print(f"[probe] pool_recall@{s} = {r:.3f}")
        last = r
        chosen = s
        if r >= targets[0]:
            break
    print(f"[auto] using bm25_size={chosen} (pool_recall≈{last:.3f})")
    return chosen, last

def split_train_valid(pairs, valid_frac=0.2, min_q=1):
    """Simple per-id split. Keeps at least one query for train when possible."""
    train, valid = {}, {}
    for sid, qs in pairs.items():
        qs = [q for q in qs if q and isinstance(q, str)]
        if not qs:
            continue
        n_valid = max(1, int(len(qs) * valid_frac))
        valid[sid] = qs[:n_valid]
        rest = qs[n_valid:]
        if rest:
            train[sid] = rest
        elif len(qs) >= 2:
            train[sid] = qs[1:]
        else:
            # if only one query, keep it for train
            train[sid] = qs
    return train, valid

def evaluate(os_client, index, pairs, model, *, top_k=10):
    hr, mrr, n = 0.0, 0.0, 0
    for gold, queries in pairs.items():
        for q in queries:
            hits = retrieve_with_ltr(os_client, index, q, model, cfg=RET)
            ids  = [h["_source"].get("show_id","") for h in hits]
            hr  += hit_rate_at_k(ids, gold, k=top_k)
            mrr += mrr_at_k(ids, gold, k=top_k)
            n   += 1
    return {"hit_rate": hr / max(1, n), "mrr": mrr / max(1, n), "n": n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://localhost:9200")
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--pairs", required=True, help="Path to pairs.json")
    ap.add_argument("--model_out", default=MODEL_PATH)
    args = ap.parse_args()

    cfg = Cfg(
        url=args.url,
        index=args.index
    )
    os_client = make_client(cfg)
    with open(args.pairs, "r") as f:
        pairs = json.load(f)

    print(f"[main] using OpenSearch client: {cfg.url} index={cfg.index} and starting pool recall")

    # 1) Ensure candidate pool recall is healthy; auto-tune bm25_size
    bm25_size, pool_r = auto_bm25_size(os_client, args.index, pairs)
    RET["bm25_size"] = bm25_size

    print(f"Starting split for validation")

    # 2) Split for validation
    train_pairs, valid_pairs = split_train_valid(pairs, valid_frac=0.2)

    print("Starting training ")
    # 3) Train LTR
    print("[train] building features & training LightGBM…")
    model = train_ltr(os_client, args.index, train_pairs, cfg=RET)
    save_model(model, args.model_out)
    print(f"[train] saved -> {args.model_out}")

    # 4) Evaluate on validation
    print("[eval] scoring validation set…")
    model = load_model(args.model_out)
    metrics = evaluate(os_client, args.index, valid_pairs, model, top_k=RET["top_k"])
    print(f"[eval] HR@{RET['top_k']}={metrics['hit_rate']:.3f}  MRR@{RET['top_k']}={metrics['mrr']:.3f}  n={metrics['n']}")

    # 5) (Optional) Train on full set and save final
    # model_full = train_ltr(os_client, args.index, pairs, cfg=RET)
    # save_model(model_full, args.model_out)

if __name__ == "__main__":
    main()
