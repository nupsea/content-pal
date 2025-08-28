#!/usr/bin/env python3
"""
Demonstration of the modular Content-Pal system
"""

import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.search import ContentSearchSystem
from modules.rag import AdaptiveRetriever
from modules.evaluation import SearchEvaluator, GroundTruthGenerator


def demo_basic_search():
    """Demonstrate basic search functionality"""
    print("\n" + "="*60)
    print("BASIC SEARCH DEMO")
    print("="*60)
    
    # Test with MinSearch (lightweight, no dependencies)
    print("\n1. Testing MinSearch Backend:")
    try:
        search_system = ContentSearchSystem(backend_type="minsearch")
        
        # Check if we have data
        data_path = "data/netflix_titles_cleaned.csv"
        if os.path.exists(data_path):
            search_system.index_data(csv_path=data_path)
            
            # Test searches
            test_queries = [
                "The Matrix",
                "Will Smith movies",
                "comedy movies"
            ]
            
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                results = search_system.search(query)
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results[:3], 1):
                    print(f"  {i}. {result.title} ({result.content_type})")
        else:
            print(f"Data file not found at {data_path}")
        
    except ImportError as e:
        print(f"MinSearch not available: {e}")
    
    # Test with OpenSearch (full-featured)
    print("\n2. Testing OpenSearch Backend:")
    try:
        search_system = ContentSearchSystem(backend_type="opensearch")
        print(f"OpenSearch backend initialized: {search_system.get_info()}")
        
    except ImportError as e:
        print(f"OpenSearch not available: {e}")


def demo_adaptive_retrieval():
    """Demonstrate adaptive RAG retrieval"""
    print("\n" + "="*60)
    print("ADAPTIVE RETRIEVAL DEMO")
    print("="*60)
    
    try:
        # Create adaptive retriever (falls back to MinSearch if OpenSearch unavailable)
        retriever = AdaptiveRetriever()
        
        # Check if we have data
        data_path = "data/netflix_titles_cleaned.csv"
        if os.path.exists(data_path):
            retriever.index_data(csv_path=data_path)
            
            # Test different query types
            test_queries = [
                ("The Matrix", "Expected: Title search"),
                ("Will Smith movies", "Expected: Actor search"),
                ("romantic comedies", "Expected: Genre search"),
                ("movies from 2020", "Expected: Year search"),
                ("Christopher Nolan films", "Expected: Director search"),
                ("sci-fi action movies", "Expected: Combined search")
            ]
            
            for query, expected in test_queries:
                print(f"\nQuery: '{query}' ({expected})")
                result = retriever.retrieve(query, top_k=5)
                
                intent = result.get('query_intent')
                if intent:
                    print(f"Classified as: {intent.intent_type.value} (confidence: {intent.confidence:.2f})")
                    print(f"Strategy used: {result.get('strategy_used')}")
                
                hits = result.get('hits', [])
                print(f"Found {len(hits)} results:")
                for i, hit in enumerate(hits[:3], 1):
                    source = hit.get('_source', {})
                    title = source.get('title', 'Unknown')
                    content_type = source.get('type', 'Unknown')
                    print(f"  {i}. {title} ({content_type})")
        else:
            print(f"Data file not found at {data_path}")
    
    except Exception as e:
        print(f"Error in adaptive retrieval demo: {e}")


def demo_evaluation_system():
    """Demonstrate evaluation system"""
    print("\n" + "="*60)
    print("EVALUATION SYSTEM DEMO")
    print("="*60)
    
    try:
        # Generate small ground truth for demo
        data_path = "data/netflix_titles_cleaned.csv"
        if not os.path.exists(data_path):
            print(f"Data file not found at {data_path}")
            return
        
        gt_generator = GroundTruthGenerator()
        ground_truth = gt_generator.generate_ground_truth(
            csv_path=data_path, 
            sample_size=100
        )
        
        # Create evaluation subset
        eval_subset = gt_generator.create_evaluation_subset(ground_truth, subset_size=20)
        
        # Create search system for evaluation
        try:
            search_system = ContentSearchSystem(backend_type="minsearch")
            search_system.index_data(csv_path=data_path)
            
            # Create evaluator
            evaluator = SearchEvaluator(search_system)
            
            results = evaluator.evaluate_queries(
                eval_subset, 
                max_queries=50,
                workers=2
            )
            
            # Print summary
            evaluator.print_summary(results)
            
        except ImportError:
            print("MinSearch not available for evaluation demo")
    
    except Exception as e:
        print(f"Error in evaluation demo: {e}")


def demo_system_comparison():
    """Demonstrate system comparison"""
    print("\n" + "="*60)
    print("SYSTEM COMPARISON DEMO")
    print("="*60)
    
    try:
        data_path = "data/netflix_titles_cleaned.csv"
        if not os.path.exists(data_path):
            print(f"Data file not found at {data_path}")
            return
        
        # Create test queries using real Netflix data
        test_ground_truth = {
            's8415': ['the matrix', 'matrix movie'],  # The Matrix
            's592': ['will smith seven pounds', 'seven pounds'],  # Seven Pounds with Will Smith
            's829': ['will smith collateral beauty', 'collateral beauty']  # Collateral Beauty with Will Smith
        }
        
        try:
            # Test basic search
            basic_system = ContentSearchSystem(backend_type="minsearch")
            basic_system.index_data(csv_path=data_path)
            basic_evaluator = SearchEvaluator(basic_system)
            
            basic_results = basic_evaluator.evaluate_queries(
                test_ground_truth, 
                max_queries=10,
                workers=1
            )
            
            # Test adaptive system  
            adaptive_system = AdaptiveRetriever()
            adaptive_system.index_data(csv_path=data_path)
            adaptive_evaluator = SearchEvaluator(adaptive_system)
            
            adaptive_results = adaptive_evaluator.evaluate_queries(
                test_ground_truth,
                max_queries=10, 
                workers=1
            )
            
            # Compare results
            print(f"\nCOMPARISON RESULTS:")
            print(f"{'System':<15} {'HR@1':<8} {'HR@5':<8} {'MRR@10':<8}")
            print("-" * 40)
            
            basic_metrics = basic_results['overall_metrics']
            adaptive_metrics = adaptive_results['overall_metrics']
            
            print(f"{'Basic':<15} {basic_metrics.get('hit_rate_at_1', 0):<8.4f} "
                  f"{basic_metrics.get('hit_rate_at_5', 0):<8.4f} {basic_metrics.get('mrr_at_10', 0):<8.4f}")
            print(f"{'Adaptive':<15} {adaptive_metrics.get('hit_rate_at_1', 0):<8.4f} "
                  f"{adaptive_metrics.get('hit_rate_at_5', 0):<8.4f} {adaptive_metrics.get('mrr_at_10', 0):<8.4f}")
            
        except ImportError:
            print("Required dependencies not available for comparison")
    
    except Exception as e:
        print(f"Error in comparison demo: {e}")


def main():
    """Run all demonstrations"""
    print("Content-Pal Modular System Demonstration")
    print("This demo showcases the new lean, extensible architecture")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run demos
    demo_basic_search()
    demo_adaptive_retrieval()
    demo_evaluation_system()
    demo_system_comparison()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nThe modular system provides:")
    print("* Clean separation of concerns (search, RAG, LLM, evaluation)")
    print("* Pluggable backends (MinSearch, OpenSearch)")
    print("* Adaptive query processing")
    print("* Comprehensive evaluation framework")
    print("* Easy extensibility for future components")
    print("\nNext steps:")
    print("- Add LLM integrations in modules/llm/")
    print("- Implement MCP (Model Context Protocol) support")
    print("- Add more search backends")
    print("- Extend evaluation metrics")


if __name__ == "__main__":
    main()