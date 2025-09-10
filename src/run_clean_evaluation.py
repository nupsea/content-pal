#!/usr/bin/env python3
"""
Clean comprehensive evaluation with only essential high-performing systems
"""

import sys
import json
from pathlib import Path
import pandas as pd


# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.search import ContentSearchSystem
from modules.search.strategic_search import StrategicSearchSystem, StrategicSearchConfig
from modules.search.pretrained_semantic_search import PretrainedSemanticSearch
# from modules.search.pts import PretrainedSemanticSearch as PTS
from modules.search.aware_pts import SchemaAwareSemanticSearch as PTS
from modules.search.enriched_semantic_search import EnrichedSemanticSearch
from modules.search.simple_effective_search import SimpleEffectiveSearch
from modules.search.ultraboost_search import UltraBoostSearchSystem
from modules.search.cross_encoder_reranker import get_cross_encoder_reranker, prepare_cross_encoder_reranking
# Legacy RAG import removed - not used in current evaluation
from modules.evaluation import SearchEvaluator, GroundTruthGenerator


def run_clean_evaluation():
    """Run evaluation with only high-performing systems"""
    
    # Suppress verbose logging during evaluation
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    # Suppress print statements from search modules during evaluation
    import builtins
    original_print = builtins.print
    
    def filtered_print(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        # Only show important messages, suppress search system noise
        if not any(noise in message for noise in ['**', '[ULTRA]', '[SIMPLE]', '!!']):
            original_print(*args, **kwargs)
    
    builtins.print = filtered_print
    
    try:
        print("CLEAN COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Load ground truth  
        possible_paths = [
            "data/ground_truth/new_ground_truth.json",
            "notebooks/ground_truth.json", 
            str(Path(__file__).parent.parent / "data/ground_truth/new_ground_truth.json")
        ]
        
        gt_file = None
        for path in possible_paths:
            if Path(path).exists():
                gt_file = path
                break
        
        if not gt_file:
            print("Ground truth file not found in any of these locations:")
            for path in possible_paths:
                print(f"   {Path(path).absolute()}")
            return
        
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)
        
        print(f"[OK] Loaded ground truth: {len(ground_truth)} unique assets")
        
        # Create evaluation subset (limit for reasonable runtime)
        import random
        random.seed(42)  # Reproducible results
        
        asset_ids = list(ground_truth.keys())
        random.shuffle(asset_ids)
        subset_size = min(200, len(asset_ids))
        eval_subset = {aid: ground_truth[aid] for aid in asset_ids[:subset_size]}
        
        print(f"[OK] Created evaluation subset: {subset_size} assets")
        
        # High-performing systems configuration
        configurations = [
            {
                "name": "SchemaAwareSemanticSearch",
                "system_type": "pts", 
                "backend": "minsearch",
                "csv_path": "data/netflix_titles_enriched_full.csv"
            },
            {
                "name": "SimpleEffectiveSearch",
                "system_type": "simple_effective", 
                "backend": "minsearch",
                "csv_path": "data/netflix_titles_enriched_full.csv"
            },
            {
                "name": "UltraBoostSearch", 
                "system_type": "ultra_boost",
                "backend": "minsearch",
                "csv_path": "data/netflix_titles_enriched_full.csv"
            }
        ]
        
        results = {}
        
        for config in configurations:
            print(f"\nTesting: {config['name']}")
            
            try:
                csv_path = config.get("csv_path", "data/netflix_titles_enriched_full.csv")
                
                if config["system_type"] == "pts":
                    # PTS system using improved pre-trained models
                    # system = PTS(backend_type=config["backend"], per_head_k=300, max_candidates_cap=1500, alpha_semantic=0.97)

                    # Aware PTS
                    system = PTS(
                        backend_type=config["backend"],
                        per_head_k=250,
                        max_candidates_cap=1500,
                        alpha_semantic=0.95,
                        prf_m=40,
                        prf_top_facets=3,
                        use_dense_head=False,    # keep False for now; you can try True later
                        dense_k=100
                    )
                    system.index_data(csv_path=csv_path)

                    coverage = candidate_coverage(system, eval_subset, csv_path, 600) * 100
                    ce_win = ce_win_rate(system, eval_subset, csv_path, 10) * 100
                    print(f"Candidate coverage: {coverage:.1f}% | CE win-rate: {ce_win:.1f}%")

                    evaluator = SearchEvaluator(system)
                    
                elif config["system_type"] == "simple_effective":
                    # Simple Effective Search System
                    system = SimpleEffectiveSearch(backend_type=config["backend"])
                    system.index_data(csv_path=csv_path)
                    
                    # Create wrapper for SearchEvaluator compatibility  
                    class SimpleEffectiveSearchWrapper:
                        def __init__(self, search_system):
                            self.search_system = search_system
                        
                        def search(self, query: str, config=None):
                            top_k = 50  # Default for evaluation
                            return self.search_system.search(query, top_k=top_k)
                    
                    wrapped_system = SimpleEffectiveSearchWrapper(system)
                    evaluator = SearchEvaluator(wrapped_system)
                
                elif config["system_type"] == "ultra_boost":
                    # UltraBoost system (now in separate module)
                    system = UltraBoostSearchSystem(backend_type=config["backend"])
                    system.index_data(csv_path=csv_path)
                    evaluator = SearchEvaluator(system)
                
                else:
                    print(f"Unknown system type: {config['system_type']}")
                    continue
                
                # Run evaluation
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=500,
                    workers=4,
                    top_k=50,
                    use_reranking=False
                )
                
                results[config["name"]] = eval_results
                
                # Print quick metrics summary right after evaluation
                summary = eval_results.get("summary", {})
                overall_metrics = eval_results.get("overall_metrics", {})
                hr_10 = overall_metrics.get("hit_rate_at_10", 0.0) * 100
                mrr_10 = overall_metrics.get("mrr_at_10", 0.0) * 100
                total_queries = summary.get("total_queries", 0)
                print(f"→ {config['name']}: HR@10={hr_10:.1f}% | MRR@10={mrr_10:.1f}% | Queries={total_queries}")
                
            except Exception as e:
                print(f"{config['name']} system failed: {e}")
                results[config["name"]] = {"error": str(e)}
        
        # Print comparison results
        print(f"\n{'='*60}")
        print("COMPREHENSIVE EVALUATION RESULTS")
        print(f"{'='*60}")
        
        for name, result in results.items():
            if "error" in result:
                print(f"\n{name}: ERROR - {result['error']}")
                continue
                
            summary = result.get("summary", {})
            overall_metrics = result.get("overall_metrics", {})
            
            total_queries = summary.get("total_queries", 0)
            successful_queries = summary.get("successful_queries", 0)
            hr_10 = overall_metrics.get("hit_rate_at_10", 0.0) * 100
            mrr_10 = overall_metrics.get("mrr_at_10", 0.0) * 100
            avg_time = summary.get("avg_query_time_ms", 0.0)
            
            print(f"\n{name}:")
            print(f"  Total Queries: {total_queries}")
            print(f"  Successful Queries: {successful_queries}")
            print(f"  Hit Rate @ 10: {hr_10:.1f}%")
            print(f"  MRR @ 10: {mrr_10:.1f}%")
            print(f"  Avg Query Time: {avg_time:.1f}ms")
            
        
        # Save results
        output_file = "clean_evaluation_results.json"
        with open(output_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for name, result in results.items():
                if isinstance(result, dict) and "error" not in result:
                    serializable_results[name] = {
                        "summary": result.get("summary", {}),
                        "overall_metrics": result.get("overall_metrics", {}),
                        "system_name": name
                    }
                else:
                    serializable_results[name] = result
                    
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        return results
    
    finally:
        # Restore original print function
        builtins.print = original_print


# After you build your system and load ground truth:
# ---- coverage / win-rate helpers (drop-in) ----

def _build_catalog_maps(csv_path: str):
    df = pd.read_csv(csv_path, encoding="latin-1").fillna("")
    show_ids = {str(x).strip().lower() for x in df["show_id"].astype(str)}
    title2id = {str(t).strip().lower(): str(s) for t, s in zip(df["title"], df["show_id"])}
    return show_ids, title2id

def _resolve_to_show_ids(obj, key, show_ids_lc, title2id):
    """
    Return a list of canonical show_ids for one GT entry.
    Tries (1) explicit show_id, (2) title, (3) asset key itself if it looks like a show_id.
    """
    out = []

    # obj may be dict with metadata
    if isinstance(obj, dict):
        cand = obj.get("show_id") or obj.get("_id") or obj.get("id")
        if cand and str(cand).strip().lower() in show_ids_lc:
            out.append(str(cand).strip())
        else:
            title = obj.get("title")
            if title:
                sid = title2id.get(str(title).strip().lower())
                if sid: out.append(sid)

    # last resort: try the GT key itself (asset id) as a show_id
    if key and str(key).strip().lower() in show_ids_lc:
        out.append(str(key).strip())

    return list(dict.fromkeys(out))  # dedupe

def _iter_query_showids_from_gt(gt, csv_path: str):
    """
    Yields (query_string, [relevant_show_ids]) pairs for either GT shape:
      1) asset-centric: {asset_id: {"queries":[...], ...}}
      2) query-centric: {"query string": ["s123", "s456"]}
    """
    show_ids_lc, title2id = _build_catalog_maps(csv_path)

    if isinstance(gt, dict):
        sample = next(iter(gt.values()))
        if isinstance(sample, dict) and "queries" in sample:
            # asset-centric with nested dict format: {asset_id: {"queries": [...], ...}}
            for asset_id, obj in gt.items():
                rel_ids = _resolve_to_show_ids(obj, asset_id, show_ids_lc, title2id)
                for q in obj.get("queries", []):
                    yield q, rel_ids
        elif isinstance(sample, list):
            # asset-centric with list format: {asset_id: ["query1", "query2", ...]}
            for asset_id, queries in gt.items():
                if asset_id.lower() in show_ids_lc:
                    rel_ids = [asset_id]
                else:
                    rel_ids = []
                for q in queries:
                    yield q, rel_ids
        else:
            # query-centric: {"query string": ["show_id1", "show_id2", ...]}
            for q, rel in gt.items():
                rel_ids = []
                for x in rel:
                    x = str(x).strip()
                    if x.lower() in show_ids_lc:
                        rel_ids.append(x)
                    else:
                        sid = title2id.get(x.lower())
                        if sid: rel_ids.append(sid)
                yield q, list(dict.fromkeys(rel_ids))
    elif isinstance(gt, list):
        # list of {"query":..., "relevant":[...]}
        show_ids_lc, title2id = _build_catalog_maps(csv_path)
        for item in gt:
            q = item["query"]
            rel_ids = []
            for x in item.get("relevant", []):
                x = str(x).strip()
                if x.lower() in show_ids_lc:
                    rel_ids.append(x)
                else:
                    sid = title2id.get(x.lower())
                    if sid: rel_ids.append(sid)
            yield q, list(dict.fromkeys(rel_ids))
    else:
        raise ValueError("Unknown GT structure")

def candidate_coverage(system, gt, csv_path, cand_cap=600):
    ok = total = 0
    for q, rel in _iter_query_showids_from_gt(gt, csv_path):
        cand_ids = {str(c.id).strip().lower() for c in system._candidates(q)[:cand_cap]}
        rel_lc = {r.strip().lower() for r in rel}
        hit = bool(cand_ids & rel_lc)
        ok += 1 if hit else 0
        total += 1
    return ok / max(total, 1)

def ce_win_rate(system, gt, csv_path, topK=10):
    improved = total = 0
    for q, rel in _iter_query_showids_from_gt(gt, csv_path):
        rel_lc = {r.strip().lower() for r in rel}
        cands = system._candidates(q)
        if not cands: 
            continue
        base_top = [str(c.id).strip().lower() for c in cands[:topK]]
        ce_top   = [str(c.id).strip().lower() for c in system._ce_rerank(q, cands, topK)]
        # only count queries where a relevant doc is somewhere in candidates
        if any(cid in rel_lc for cid in {str(c.id).strip().lower() for c in cands}):
            def rr(lst):
                for i, d in enumerate(lst, 1):
                    if d in rel_lc: return 1.0 / i
                return 0.0
            if rr(ce_top) > rr(base_top):
                improved += 1
            total += 1
    return improved / max(total, 1)




if __name__ == "__main__":
    run_clean_evaluation()