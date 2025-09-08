# Scripts Directory

This directory contains all utility scripts organized by functionality.

## 📁 **Structure**

### `data_generation/`
Scripts for creating and processing datasets:
- `create_comprehensive_no_api_dataset.py` - Generate enriched dataset without external APIs
- `create_proper_enriched_dataset.py` - Create properly structured enriched dataset
- `generate_full_enriched_dataset.py` - Generate complete enriched dataset
- `fix_enriched_dataset.py` - Fix issues in existing enriched datasets

### `ground_truth/`
Scripts for generating and validating ground truth data:
- `create_realistic_ground_truth.py` - Generate realistic ground truth queries
- `generate_comprehensive_ground_truth.py` - Create comprehensive evaluation ground truth
- `generate_realistic_ground_truth.py` - Generate realistic query-document pairs
- `validate_comprehensive_ground_truth.py` - Validate ground truth quality

### `evaluation/`
Scripts for running evaluations and comparisons:
- `run_advanced_evaluation.py` - Run advanced system evaluations
- `run_evaluation_with_new_gt.py` - Evaluate with new ground truth data
- `run_focused_evaluation.py` - Run focused evaluation on specific systems

### `models/`
Scripts for training and managing models:
- `train_semantic_model.py` - Train semantic search models
- `create_learned_query_optimizer.py` - Create learned query optimization models
- `learned_query_optimizer.pkl` - Trained query optimizer model

## 🚀 **Usage**

All scripts should be run from the project root directory:

```bash
# Data generation
python scripts/data_generation/create_proper_enriched_dataset.py

# Ground truth generation
python scripts/ground_truth/generate_comprehensive_ground_truth.py

# Evaluation
python scripts/evaluation/run_advanced_evaluation.py

# Model training
python scripts/models/train_semantic_model.py
```

## 📋 **Dependencies**

Most scripts require the main project environment:
```bash
pipenv install
pipenv shell
```