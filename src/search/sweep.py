# sweep.py
import itertools, json, subprocess, sys
import os

index = "netflix_assets_v4"
pairs = "../../notebooks/ground_truth.json"

bm25_fin = [300, 350, 400]
annk     = [300, 400, 500]
exp_boost= [0.5, 0.6, 0.7]

for bf, ak, eb in itertools.product(bm25_fin, annk, exp_boost):
    env = {"HYB_BM25_FINAL": str(bf), "HYB_ANN_K": str(ak), "HYB_EXP_BOOST": str(eb)}
    # your eval_parallel.py can read these envs and pass into hybrid_search/bm25_prf
    res = subprocess.check_output([
        sys.executable, "eval_parallel.py",
        "--index", index,
        "--pairs", pairs,
        "--top_k", "10",
        "--pool_top_k", "80",
        "--workers", "16",
        "--max_pairs", "500"
    ], env={**os.environ, **env})
    print(bf, ak, eb, res.decode().strip())
