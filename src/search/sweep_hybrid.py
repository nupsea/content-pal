#!/usr/bin/env python3
from __future__ import annotations
import os, json, subprocess, sys, itertools, time
from pathlib import Path

PY = sys.executable
EVAL = Path(__file__).with_name("eval_search_parallel.py")  # or your eval filename

INDEX = os.getenv("OS_INDEX", "netflix_assets_v4")
PAIRS = os.getenv("GT_PATH", "../../notebooks/ground_truth.json")

# grids (keep small)
BM25_FINAL = [300, 350, 400]
ANN_K      = [300, 400, 500]
EXP_BOOST  = [0.5, 0.6, 0.7]
RRF_K      = [40.0, 60.0, 80.0]
W_BM       = [1.0, 1.2]
W_ANN      = [1.0, 1.2]

def run_eval(env_overrides: dict) -> dict:
    env = {**os.environ, **{k: str(v) for k,v in env_overrides.items()}}
    # fast parallel eval on a sample; adjust as needed
    cmd = [
        PY, str(EVAL),
        "--index", INDEX,
        "--pairs", PAIRS,
        "--top_k", "10",
        "--pool_top_k", "80",
        "--workers", "12",
        "--max_pairs", "500",
        # "--use_cross_encoder",  # uncomment to include CE (slower, higher MRR)
    ]
    out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT)
    # eval script prints a dict; parse safely
    s = out.decode().strip().splitlines()[-1]
    return json.loads(s.replace("'", '"'))

def main():
    results = []
    start = time.time()
    for bf, ak, eb, rk, wb, wa in itertools.product(BM25_FINAL, ANN_K, EXP_BOOST, RRF_K, W_BM, W_ANN):
        cfg = {
            "HYB_BM25_FINAL": bf,
            "HYB_ANN_K": ak,
            "HYB_EXP_BOOST": eb,
            "HYB_RRF_K": rk,
            "HYB_W_BM": wb,
            "HYB_W_ANN": wa,
        }
        try:
            metrics = run_eval(cfg)
            results.append((metrics, cfg))
            print({**metrics, **cfg})
        except subprocess.CalledProcessError as e:
            print("ERR:", e.output.decode())
    dur = time.time() - start
    print(f"\nTried {len(results)} configs in {dur/60:.1f} min")

    # sort by MRR then HR
    results.sort(key=lambda x: (x[0]["mrr"], x[0]["hit_rate"]), reverse=True)
    print("\nTop 5:")
    for (m, c) in results[:5]:
        print(m, c)

if __name__ == "__main__":
    main()
