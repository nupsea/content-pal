# Data Directory

This directory contains all datasets, ground truth, and result files.

## 📁 **Structure**

### **Root Level - Main Datasets**
- `netflix_titles.csv` - Original Netflix dataset
- `netflix_titles_cleaned.csv` - Cleaned version of original dataset
- `netflix_titles_enriched.csv` - Basic enriched dataset
- `netflix_titles_enriched_full.csv` - **Main dataset** (used by search systems)
- `netflix_titles_expanded.csv` - Dataset with expanded text
- `netflix_sample_enriched.csv` - Sample enriched dataset for testing
- `netflix_no_api_enriched_sample.csv` - No-API enriched sample

### `ground_truth/`
Ground truth data for evaluation:
- `new_ground_truth.json` - **Primary ground truth** (1000 assets)
- `realistic_ground_truth.json` - Realistic query-document pairs
- `realistic_evaluation_subset.json` - Subset for focused evaluation

### `results/`
Analysis and evaluation results:
- `advanced_success_analysis.json` - Analysis of system performance patterns

### `scraped_cache/`
Cached external data from web scraping:
- `imdb_basic_*.json` - IMDB metadata cache
- `wikipedia_*.json` - Wikipedia content cache

## 🎯 **Key Files**

### **Primary Dataset**
```
data/netflix_titles_enriched_full.csv
```
- **7,370 movies/shows** with semantic enrichment
- **16.2 average tags** per item
- Used by all search systems (SimpleEffective, UltraBoost)

### **Primary Ground Truth**
```
data/ground_truth/new_ground_truth.json
```
- **1,000 unique assets** with curated queries
- **5 queries per asset** on average
- Used for comprehensive evaluation (target: >50% HR@10)

## 📊 **Dataset Statistics**

| Dataset | Items | Features | Usage |
|---------|-------|----------|-------|
| Original | 8,807 | Basic metadata | Source |
| Cleaned | 7,787 | Cleaned metadata | Processing |
| Enriched Full | 7,370 | + Semantic tags | **Production** |
| Ground Truth | 1,000 | + Query pairs | **Evaluation** |

## 🔄 **Data Pipeline**

```
netflix_titles.csv
    ↓ (clean)
netflix_titles_cleaned.csv
    ↓ (enrich with semantic tags)
netflix_titles_enriched_full.csv
    ↓ (generate query pairs)
ground_truth/new_ground_truth.json
```