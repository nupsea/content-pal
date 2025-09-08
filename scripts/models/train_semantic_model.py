#!/usr/bin/env python3
"""
Train the learned semantic search model using ground truth data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from modules.search.learned_semantic_search import LearnedSemanticSearch

def train_semantic_model():
    """Train semantic model on ground truth data"""
    
    print("🎓 TRAINING LEARNED SEMANTIC SEARCH MODEL")
    print("=" * 60)
    
    # Initialize system
    system = LearnedSemanticSearch(backend_type="minsearch")
    
    # Index data
    print("📚 Indexing data...")
    system.index_data(csv_path="data/netflix_titles_cleaned.csv")
    
    # Train from ground truth
    print("🚀 Training semantic model...")
    success = system.train_from_ground_truth(
        ground_truth_path="new_ground_truth.json",
        num_epochs=2  # Start with few epochs for testing
    )
    
    if success:
        print("✅ Training completed successfully!")
        
        # Test the trained model
        print("\n🧪 Testing trained model...")
        test_queries = [
            "feel good movies about faith",
            "documentaries about chess", 
            "psychological movies with Amber Midthunder"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = system.search(query, top_k=5)
            
            for i, result in enumerate(results):
                print(f"  {i+1}. '{result.title}' (Score: {result.score:.3f})")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    train_semantic_model()