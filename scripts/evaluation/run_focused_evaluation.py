#!/usr/bin/env python3
"""
Focused evaluation of key fixed systems
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from modules.search import ContentSearchSystem
from modules.search.enhanced_search import EnhancedSearchSystem
from modules.search.strategic_search import StrategicSearchSystem, StrategicSearchConfig
from modules.search.pretrained_semantic_search import PretrainedSemanticSearch
from modules.search.enriched_semantic_search import EnrichedSemanticSearch
from modules.search.cross_encoder_reranker import get_cross_encoder_reranker, prepare_cross_encoder_reranking
from modules.rag_old import AdaptiveRetriever
from modules.evaluation import SearchEvaluator


def run_focused_evaluation():
    """Run evaluation on key systems only"""
    print("FOCUSED EVALUATION - KEY SYSTEMS")
    print("=" * 60)
    
    # Load ground truth
    possible_paths = [
        "../new_ground_truth.json",
        "new_ground_truth.json", 
        str(Path(__file__).parent.parent / "new_ground_truth.json")
    ]
    
    gt_file = None
    for path in possible_paths:
        if Path(path).exists():
            gt_file = path
            break
    
    if not gt_file:
        print("❌ Ground truth file not found")
        return
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"✅ Loaded ground truth: {len(ground_truth)} unique assets")
    
    # Create small evaluation subset  
    import random
    random.seed(42)
    asset_ids = list(ground_truth.keys())
    random.shuffle(asset_ids)
    subset_size = min(50, len(asset_ids))  # Small subset for quick test
    eval_subset = {aid: ground_truth[aid] for aid in asset_ids[:subset_size]}
    
    print(f"✅ Created evaluation subset: {subset_size} assets")
    
    # Test configurations - maximum performance systems
    configurations = [
        {
            "name": "Adaptive_UltraOptimized_v1",
            "system_type": "adaptive_enhanced",
            "semantic_weight": 0.97,
            "recall_multiplier": 12
        },
        {
            "name": "Adaptive_UltraOptimized_v2", 
            "system_type": "adaptive_enhanced",
            "semantic_weight": 0.99,
            "recall_multiplier": 15
        },
        {
            "name": "Adaptive_UltraOptimized_v3",
            "system_type": "adaptive_enhanced", 
            "semantic_weight": 0.999,
            "recall_multiplier": 20
        }
    ]
    
    results = {}
    csv_path = "data/netflix_titles_enriched_full.csv"
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        try:
            if config["system_type"] == "enhanced":
                system = EnhancedSearchSystem(backend_type="minsearch")
                system.index_data(csv_path=csv_path)
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=100,
                    workers=2,
                    top_k=50,
                    use_reranking=False
                )
                
            elif config["system_type"] == "strategic":
                strategic_system = StrategicSearchSystem(backend_type="minsearch")
                strategic_system.index_data(csv_path=csv_path)
                evaluator = SearchEvaluator(strategic_system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=100,
                    workers=2,
                    top_k=50,
                    use_reranking=False
                )
                
            elif config["system_type"] == "adaptive_enhanced":
                # Adaptive search enhanced with cross-encoder reranking
                from modules.rag_old import AdaptiveRetriever
                
                base_system = AdaptiveRetriever(backend_type="minsearch")
                base_system.index_data(csv_path=csv_path)
                
                # Prepare cross-encoder reranking
                prepare_cross_encoder_reranking(csv_path)
                
                # Create wrapper that adds cross-encoder reranking
                class AdaptiveWithCrossEncoder:
                    def __init__(self, base_system, semantic_weight=0.95, recall_multiplier=12):
                        self.base_system = base_system
                        self.reranker = get_cross_encoder_reranker()
                        self.semantic_weight = semantic_weight
                        self.recall_multiplier = recall_multiplier
                    
                    def retrieve(self, query: str, top_k: int = 50, use_reranking: bool = False, **kwargs):
                        try:
                            base_results = self.base_system.retrieve(
                                query, 
                                top_k=min(600, top_k * self.recall_multiplier), 
                                use_reranking=False,
                                **kwargs
                            )
                            hits = base_results.get('hits', [])
                            
                            metadata = {
                                'query_intent': base_results.get('query_intent'),
                                'strategy_used': base_results.get('strategy_used', 'adaptive_cross_encoder'),
                                'total_hits': base_results.get('total_hits', len(hits))
                            }
                        except Exception as e:
                            return {
                                'hits': [],
                                'query_intent': None,
                                'strategy_used': 'error',
                                'total_hits': 0
                            }
                        
                        # Convert to SearchResult format for cross-encoder
                        from modules.search.core import SearchResult
                        candidates = []
                        for hit in hits:
                            source = hit.get('_source', {})
                            result = SearchResult(
                                id=source.get('show_id', ''),
                                title=source.get('title', ''),
                                score=hit.get('_score', 1.0),
                                content_type=source.get('type', ''),
                                metadata=source
                            )
                            candidates.append(result)
                        
                        # Apply cross-encoder reranking with configurable semantic weighting for >50% performance
                        if candidates:
                            enhanced_results = self.reranker.rerank(query, candidates, top_k=top_k, semantic_weight=self.semantic_weight)
                            reranked_hits = []
                            for result in enhanced_results:
                                reranked_hits.append({
                                    '_source': result.metadata,
                                    '_score': result.score
                                })
                            
                            return {
                                'hits': reranked_hits,
                                'query_intent': metadata.get('query_intent'),
                                'strategy_used': metadata.get('strategy_used'),
                                'total_hits': len(reranked_hits)
                            }
                        
                        return {
                            'hits': hits,
                            'query_intent': metadata.get('query_intent'),
                            'strategy_used': metadata.get('strategy_used'),
                            'total_hits': len(hits)
                        }
                
                system = AdaptiveWithCrossEncoder(
                    base_system, 
                    semantic_weight=config.get("semantic_weight", 0.95),
                    recall_multiplier=config.get("recall_multiplier", 12)
                )
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=100,
                    workers=2,
                    top_k=50,
                    use_reranking=False
                )
                
            elif config["system_type"] == "pretrained_semantic":
                pretrained_system = PretrainedSemanticSearch(backend_type="minsearch")
                pretrained_system.index_data(csv_path=csv_path)
                evaluator = SearchEvaluator(pretrained_system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=100,
                    workers=2,
                    top_k=50,
                    use_reranking=False
                )
                
            elif config["system_type"] == "ultra_boost":
                # Ultra-boosted system for maximum performance >50%
                from modules.search.enriched_semantic_search import EnrichedSemanticSearch
                from modules.search.strategic_search import StrategicSearchSystem, StrategicSearchConfig
                from modules.search.pretrained_semantic_search import PretrainedSemanticSearch
                
                class UltraBoostSystem:
                    def __init__(self):
                        # Use multiple search strategies
                        self.enriched_system = EnrichedSemanticSearch(backend_type="minsearch")
                        self.enriched_system.index_data(csv_path=csv_path)
                        
                        self.strategic_system = StrategicSearchSystem(backend_type="minsearch")
                        self.strategic_system.index_data(csv_path=csv_path)
                        
                        self.pretrained_system = PretrainedSemanticSearch(backend_type="minsearch")
                        self.pretrained_system.index_data(csv_path=csv_path)
                        
                        # Prepare cross-encoder reranking
                        prepare_cross_encoder_reranking(csv_path)
                        self.reranker = get_cross_encoder_reranker()
                    
                    def search(self, query: str, top_k: int = 50):
                        # Get results from multiple systems
                        top_k_int = int(top_k) if top_k else 50
                        
                        # System 1: Enriched semantic (high recall with tags)
                        enriched_results = self.enriched_system.search(query, top_k=min(200, top_k_int * 4))
                        
                        # System 2: Strategic (query understanding)  
                        strategic_config = StrategicSearchConfig(final_result_count=min(150, top_k_int * 3))
                        strategic_results = self.strategic_system.search(query, strategic_config)
                        
                        # System 3: Pretrained semantic (proven best performer)
                        pretrained_results = self.pretrained_system.search(query, top_k=min(100, top_k_int * 2))
                        
                        # Combine with sophisticated weighted fusion
                        combined_results = {}
                        
                        # Optimized fusion weights for >50% performance - prioritize semantic systems
                        # Add enriched results (highest weight - semantic tags are crucial)
                        for i, result in enumerate(enriched_results):
                            score = (len(enriched_results) - i) / len(enriched_results) * 0.6  # 60% weight 
                            combined_results[result.id] = result
                            result.score = score
                        
                        # Add pretrained results (strong semantic understanding)
                        for i, result in enumerate(pretrained_results):
                            score = (len(pretrained_results) - i) / len(pretrained_results) * 0.3  # 30% weight
                            if result.id not in combined_results:
                                combined_results[result.id] = result
                                result.score = score
                            else:
                                # Strong boost if found by multiple semantic systems
                                combined_results[result.id].score += score * 1.2
                        
                        # Add strategic results (query understanding diversity)
                        for i, result in enumerate(strategic_results):
                            score = (len(strategic_results) - i) / len(strategic_results) * 0.2  # 20% weight
                            if result.id not in combined_results:
                                combined_results[result.id] = result
                                result.score = score
                            else:
                                # Boost if found by multiple systems  
                                combined_results[result.id].score += score * 0.8
                        
                        # Apply ultra-high semantic reranking
                        final_candidates = list(combined_results.values())
                        if final_candidates:
                            # Sort first
                            final_candidates.sort(key=lambda x: x.score, reverse=True)
                            # Apply cross-encoder with ultra-high semantic weighting for >50%
                            final_results = self.reranker.rerank(
                                query, 
                                final_candidates[:min(250, len(final_candidates))],  # Rerank top 250 for better coverage
                                top_k=top_k_int,
                                semantic_weight=0.98  # 98% semantic for maximum performance
                            )
                        else:
                            final_results = final_candidates[:top_k_int]
                        
                        return final_results
                    
                    def retrieve(self, query: str, top_k: int = 50, **kwargs):
                        """Retrieve method for compatibility with evaluator"""
                        results = self.search(query, top_k)
                        # Convert SearchResult objects to hits format
                        hits = []
                        for result in results:
                            hits.append({
                                '_source': result.metadata,
                                '_score': result.score
                            })
                        
                        return {
                            'hits': hits,
                            'query_intent': None,
                            'strategy_used': 'ultra_boost_multi_system',
                            'total_hits': len(hits)
                        }
                
                system = UltraBoostSystem()
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=100,
                    workers=2,
                    top_k=50,
                    use_reranking=False
                )
            
            elif config["system_type"] == "ultra_semantic":
                # Ultra-semantic system optimized for maximum recall and precision
                class UltraSemanticSystem:
                    def __init__(self):
                        # Use the best semantic systems with optimized parameters
                        self.enriched_system = EnrichedSemanticSearch(backend_type="minsearch")
                        self.enriched_system.index_data(csv_path=csv_path)
                        
                        self.pretrained_system = PretrainedSemanticSearch(backend_type="minsearch")
                        self.pretrained_system.index_data(csv_path=csv_path)
                        
                        # Prepare cross-encoder reranking
                        prepare_cross_encoder_reranking(csv_path)
                        self.reranker = get_cross_encoder_reranker()
                    
                    def search(self, query: str, top_k: int = 50):
                        """Ultra-semantic search prioritizing semantic understanding"""
                        top_k_int = int(top_k) if top_k else 50
                        
                        # Get high-recall results from both semantic systems
                        enriched_results = self.enriched_system.search(query, top_k=min(300, top_k_int * 6))
                        pretrained_results = self.pretrained_system.search(query, top_k=min(200, top_k_int * 4))
                        
                        # Advanced semantic fusion - weight by semantic quality
                        combined_results = {}
                        
                        # Enriched results get highest weight (semantic tags)
                        for i, result in enumerate(enriched_results):
                            # Exponential decay for ranking position
                            position_score = 1.0 / (1.0 + i * 0.1)
                            score = position_score * 0.7  # 70% base weight
                            combined_results[result.id] = result
                            result.score = score
                        
                        # Pretrained results with complementary weighting
                        for i, result in enumerate(pretrained_results):
                            position_score = 1.0 / (1.0 + i * 0.1)
                            score = position_score * 0.5  # 50% base weight
                            if result.id not in combined_results:
                                combined_results[result.id] = result
                                result.score = score
                            else:
                                # Major boost for items found by both semantic systems
                                combined_results[result.id].score += score * 1.5
                        
                        # Apply maximum semantic reranking
                        final_candidates = list(combined_results.values())
                        if final_candidates:
                            # Sort by fusion scores first
                            final_candidates.sort(key=lambda x: x.score, reverse=True)
                            # Apply cross-encoder with near-100% semantic weighting
                            final_results = self.reranker.rerank(
                                query,
                                final_candidates[:min(300, len(final_candidates))],  # Maximum coverage
                                top_k=top_k_int,
                                semantic_weight=0.99  # 99% semantic for ultimate performance
                            )
                        else:
                            final_results = final_candidates[:top_k_int]
                        
                        return final_results
                    
                    def retrieve(self, query: str, top_k: int = 50, **kwargs):
                        """Retrieve method for compatibility with evaluator"""
                        results = self.search(query, top_k)
                        # Convert SearchResult objects to hits format
                        hits = []
                        for result in results:
                            hits.append({
                                '_source': result.metadata,
                                '_score': result.score
                            })
                        
                        return {
                            'hits': hits,
                            'query_intent': None,
                            'strategy_used': 'ultra_semantic_max_recall',
                            'total_hits': len(hits)
                        }
                
                system = UltraSemanticSystem()
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=100,
                    workers=2,
                    top_k=50,
                    use_reranking=False
                )
                
            elif config["system_type"] == "max_recall":
                # Ultimate max-recall system for >50% performance
                from modules.search.enriched_semantic_search import EnrichedSemanticSearch
                from modules.search.pretrained_semantic_search import PretrainedSemanticSearch
                from modules.rag_old import AdaptiveRetriever
                from modules.search.enhanced_search import EnhancedSearchSystem
                
                class MaxRecallUltimateSystem:
                    def __init__(self):
                        # Use ALL semantic systems for maximum coverage
                        self.enriched_system = EnrichedSemanticSearch(backend_type="minsearch")
                        self.enriched_system.index_data(csv_path=csv_path)
                        
                        self.pretrained_system = PretrainedSemanticSearch(backend_type="minsearch")
                        self.pretrained_system.index_data(csv_path=csv_path)
                        
                        self.adaptive_system = AdaptiveRetriever(backend_type="minsearch")
                        self.adaptive_system.index_data(csv_path=csv_path)
                        
                        # Enhanced search system
                        self.enhanced_system = EnhancedSearchSystem(backend_type="minsearch")
                        self.enhanced_system.index_data(csv_path=csv_path)
                        
                        # Prepare cross-encoder reranking
                        prepare_cross_encoder_reranking(csv_path)
                        self.reranker = get_cross_encoder_reranker()
                    
                    def search(self, query: str, top_k: int = 50):
                        """Maximum recall search combining ALL systems"""
                        top_k_int = int(top_k) if top_k else 50
                        
                        # Get massive recall from all systems
                        enriched_results = self.enriched_system.search(query, top_k=min(500, top_k_int * 10))
                        pretrained_results = self.pretrained_system.search(query, top_k=min(400, top_k_int * 8))
                        enhanced_results = self.enhanced_system.search(query)[:min(300, top_k_int * 6)]
                        
                        # Get adaptive results too
                        adaptive_response = self.adaptive_system.retrieve(query, top_k=min(300, top_k_int * 6))
                        adaptive_hits = adaptive_response.get('hits', [])
                        adaptive_results = []
                        for hit in adaptive_hits:
                            source = hit.get('_source', {})
                            from modules.search.core import SearchResult
                            result = SearchResult(
                                id=source.get('show_id', ''),
                                title=source.get('title', ''),
                                score=hit.get('_score', 1.0),
                                content_type=source.get('type', ''),
                                metadata=source
                            )
                            adaptive_results.append(result)
                        
                        # Ultimate fusion with quality-based weighting
                        combined_results = {}
                        
                        # Enriched gets highest weight (semantic tags + enrichment)
                        for i, result in enumerate(enriched_results):
                            score = (1.0 / (1.0 + i * 0.05)) * 0.8  # 80% weight with slower decay
                            combined_results[result.id] = result
                            result.score = score
                        
                        # Pretrained gets strong weight (proven performer)
                        for i, result in enumerate(pretrained_results):
                            score = (1.0 / (1.0 + i * 0.05)) * 0.6  # 60% weight
                            if result.id not in combined_results:
                                combined_results[result.id] = result
                                result.score = score
                            else:
                                # Big boost for semantic overlap
                                combined_results[result.id].score += score * 2.0
                        
                        # Enhanced gets medium weight
                        for i, result in enumerate(enhanced_results):
                            score = (1.0 / (1.0 + i * 0.05)) * 0.4  # 40% weight
                            if result.id not in combined_results:
                                combined_results[result.id] = result
                                result.score = score
                            else:
                                combined_results[result.id].score += score * 1.5
                                
                        # Adaptive gets lower weight but adds diversity
                        for i, result in enumerate(adaptive_results):
                            score = (1.0 / (1.0 + i * 0.05)) * 0.3  # 30% weight
                            if result.id not in combined_results:
                                combined_results[result.id] = result
                                result.score = score
                            else:
                                combined_results[result.id].score += score * 1.2
                        
                        # Apply ultimate semantic reranking
                        final_candidates = list(combined_results.values())
                        if final_candidates:
                            # Sort by fusion scores
                            final_candidates.sort(key=lambda x: x.score, reverse=True)
                            # Apply cross-encoder with maximum semantic weight
                            final_results = self.reranker.rerank(
                                query,
                                final_candidates[:min(400, len(final_candidates))],  # Use top 400 for reranking
                                top_k=top_k_int,
                                semantic_weight=0.999  # 99.9% semantic for ultimate performance
                            )
                        else:
                            final_results = final_candidates[:top_k_int]
                        
                        return final_results
                    
                    def retrieve(self, query: str, top_k: int = 50, **kwargs):
                        """Retrieve method for compatibility"""
                        results = self.search(query, top_k)
                        hits = []
                        for result in results:
                            hits.append({
                                '_source': result.metadata,
                                '_score': result.score
                            })
                        
                        return {
                            'hits': hits,
                            'query_intent': None,
                            'strategy_used': 'max_recall_ultimate_fusion',
                            'total_hits': len(hits)
                        }
                
                system = MaxRecallUltimateSystem()
                evaluator = SearchEvaluator(system)
                
                eval_results = evaluator.evaluate_queries(
                    eval_subset,
                    max_queries=100,
                    workers=2,
                    top_k=50,
                    use_reranking=False
                )
            
            results[config["name"]] = eval_results
            
            # Print quick summary
            metrics = eval_results["overall_metrics"]
            print(f"{config['name']} - HR@10: {metrics.get('hit_rate_at_10', 0):.4f}, MRR@10: {metrics.get('mrr_at_10', 0):.4f}, Time: {eval_results['summary']['avg_query_time_ms']:.1f}ms")
            
        except Exception as e:
            print(f"ERROR: {config['name']}: {e}")
            results[config["name"]] = {"error": str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("FOCUSED EVALUATION RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Configuration':<35} {'HR@10':<8} {'MRR@10':<8} {'Avg Time':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        if "error" not in result:
            metrics = result["overall_metrics"]
            summary = result["summary"]
            
            hr10 = metrics.get('hit_rate_at_10', 0)
            mrr = metrics.get('mrr_at_10', 0)
            time_ms = summary['avg_query_time_ms']
            
            print(f"{name:<35} {hr10:<8.4f} {mrr:<8.4f} {time_ms:<10.1f}")
        else:
            print(f"{name:<35} ERROR: {result['error'][:40]}")


if __name__ == "__main__":
    run_focused_evaluation()