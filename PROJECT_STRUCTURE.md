# Project Structure

Content-Pal has been reorganized into a clean, modular structure focusing on high-performing search systems.

## 🎯 **High-Level Overview**

Content-Pal is a smart streaming content assistant that helps users find movies or TV shows through advanced search capabilities. The system features high-performing search architectures:

- **SimpleEffective**: 36.0% HR@10 (our best performer)
- **UltraBoost**: 34.6% HR@10 (ensemble system)
- **Target**: >50% HR@10 (both systems approach this goal)

## 📁 **Directory Structure**

```
content-pal/
├── src/                          # Core source code
│   ├── modules/                  # Main modules
│   │   ├── search/              # ✅ Essential search systems
│   │   ├── evaluation/          # ✅ Evaluation framework  
│   │   ├── rag/                 # ✅ RAG components
│   │   └── llm/                 # LLM integrations
│   ├── preprocessing/           # Data preprocessing
│   ├── rag/                    # Additional RAG utilities
│   └── run_clean_evaluation.py # 🚀 Main evaluation script
├── data/                        # All datasets and results
│   ├── ground_truth/           # Ground truth data
│   ├── results/               # Analysis results  
│   ├── scraped_cache/         # External data cache
│   └── netflix_*.csv          # Netflix datasets
├── scripts/                    # Organized utility scripts
│   ├── data_generation/       # Dataset creation
│   ├── ground_truth/         # Ground truth generation
│   ├── evaluation/           # Evaluation runners
│   └── models/              # Model training
├── examples/                  # Usage examples
├── tools/                    # Utilities and documentation
├── notebooks/               # Jupyter analysis notebooks
└── README files            # Documentation for each folder
```

## 🔍 **Essential Search Systems**

### **SimpleEffective (Best Performer)**
- **Location**: `src/modules/search/simple_effective_search.py`
- **Performance**: 36.0% HR@10
- **Approach**: Proven field weights + cross-encoder reranking
- **Usage**: Primary production system

### **UltraBoost (Ensemble)**  
- **Components**: EnrichedSemantic + Strategic + PretrainedSemantic
- **Performance**: 34.6% HR@10
- **Approach**: Weighted ensemble with cross-encoder final reranking
- **Usage**: Advanced multi-strategy system

### **Supporting Systems**
- `enriched_semantic_search.py` - High recall with semantic tags
- `pretrained_semantic_search.py` - Cross-encoder semantic matching  
- `strategic_search.py` - Multi-strategy query understanding
- `cross_encoder_reranker.py` - Semantic reranking component

## 📊 **Data Organization**

### **Primary Dataset**
```
data/netflix_titles_enriched_full.csv (7,370 items)
- Complete enriched Netflix dataset
- 16.2 average semantic tags per item  
- Used by all search systems
```

### **Ground Truth**
```
data/ground_truth/new_ground_truth.json (1,000 assets)
- Curated query-document pairs
- 5 queries per asset average
- Primary evaluation benchmark
```

### **Results**
```
data/results/
- Performance analysis
- System comparison data
```

## 🚀 **Quick Start**

### **Run Comprehensive Evaluation**
```bash
pipenv run python src/run_clean_evaluation.py
```

### **Use SimpleEffective System**
```python
from src.modules.search.simple_effective_search import SimpleEffectiveSearch

system = SimpleEffectiveSearch(backend_type="minsearch")
system.index_data("data/netflix_titles_enriched_full.csv")
results = system.search("your query", top_k=10)
```

### **Generate New Dataset**
```bash
python scripts/data_generation/create_proper_enriched_dataset.py
```

### **Create Ground Truth**
```bash
python scripts/ground_truth/generate_comprehensive_ground_truth.py
```

## 📋 **Key Performance Metrics**

| System | HR@10 | MRR@10 | Status |
|--------|-------|--------|--------|
| SimpleEffective | 36.0% | 21.8% | ✅ Production |
| UltraBoost | 34.6% | ~20% | ✅ Advanced |
| Target | >50% | >30% | 🎯 Goal |
| Failed Advanced | 13.4% | ~8% | ❌ Removed |

## 🛠️ **Development Workflow**

1. **Data Preparation**: Use `scripts/data_generation/`
2. **Ground Truth**: Use `scripts/ground_truth/`  
3. **System Development**: Work in `src/modules/search/`
4. **Evaluation**: Run `src/run_clean_evaluation.py`
5. **Analysis**: Check results in `data/results/`

## 📚 **Documentation**

Each major directory contains its own README.md with specific details:
- `scripts/README.md` - All utility scripts
- `data/README.md` - Dataset information  
- `examples/README.md` - Usage examples
- `tools/README.md` - Development tools

## 🎯 **Next Steps**

The codebase is clean and focused on high-performing systems. To reach >50% HR@10:

1. Fine-tune SimpleEffective field weights
2. Optimize UltraBoost ensemble weights
3. Improve semantic tag quality
4. Add query preprocessing enhancements