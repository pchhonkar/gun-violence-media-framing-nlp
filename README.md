# HW5 NLP Pipeline — Media Framing Analysis

This project analyzes how four major news outlets (CNN, Fox News, NYT, WSJ) frame **victims** and **shooters** in mass shooting coverage. The pipeline uses coreference resolution, dependency parsing, sentence embeddings (SBERT), DBSCAN clustering, and chi-squared hypothesis testing to identify and compare framing patterns across outlets.

---

## 📋 Note for Graders

- **Python scripts in `src/` are the authoritative implementation**
- Run the complete pipeline with: `python -m src.run_all`
- All outputs are reproducible from the raw data in `data_100/`

---

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run complete pipeline
python -m src.run_all
```

---

## Project Structure

```
hw5-NLP/
├── data_100/                          # Raw articles (100 total, 25 per outlet)
│   ├── cnn_five_para/
│   ├── FOX_five_para/
│   ├── NYT_five_para/
│   └── WSJ_five_para/
├── outputs/
│   ├── processed/                     # CSV, JSONL, JSON data files
│   ├── figures/                       # PNG visualizations
│   └── reports/                       # Markdown documentation
├── src/
│   ├── load_articles.py               # Task 1a: Load articles
│   ├── coref_contexts.py              # Task 1b: Coreference resolution
│   ├── extract_descriptions.py        # Task 2: Extract descriptions
│   ├── embed_cluster.py               # Task 3: Embedding + clustering
│   ├── manual_eval_helpers.py         # Task 4: Manual refinement
│   ├── task5_frequency_analysis.py    # Task 5: Frequency tables
│   ├── task6_chi_square.py            # Task 6: Chi-square tests
│   ├── run_all.py                     # Run complete pipeline
│   ├── config.py                      # Configuration
│   └── utils_io.py                    # I/O utilities
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Using venv (Mac/Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Using conda

```bash
conda create -n hw5nlp python=3.10 -y
conda activate hw5nlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Verify Installation

```bash
python -c "import spacy, pandas, numpy, sklearn, scipy, matplotlib, tabulate, sentence_transformers; print('OK')"
```

---

## Running the Pipeline

### Complete Pipeline (Recommended)

```bash
python -m src.run_all                # Run all tasks (skips if outputs exist)
python -m src.run_all --force        # Force re-run all tasks
python -m src.run_all --skip-task4   # Skip manual refinement step
```

### Individual Tasks

```bash
# Task 1: Load articles + coreference resolution
python -m src.load_articles
python -m src.coref_contexts

# Task 2: Extract descriptive phrases
python -m src.extract_descriptions

# Task 3: Embed + cluster descriptions
python -m src.embed_cluster

# Task 4: Manual evaluation + refinement
python -m src.manual_eval_helpers

# Task 5: Frequency/proportion analysis
python -m src.task5_frequency_analysis

# Task 6: Chi-squared hypothesis testing
python -m src.task6_chi_square
```

---

## Output Files

### Task 1: Article Loading & Coreference

| File | Description |
|------|-------------|
| `outputs/processed/articles_master.csv` | 100 articles with metadata |
| `outputs/processed/contexts_victims.jsonl` | Victim sentences (coref-resolved) |
| `outputs/processed/contexts_shooters.jsonl` | Shooter sentences (coref-resolved) |
| `outputs/reports/task1_doc.md` | Methodology documentation |

### Task 2: Description Extraction

| File | Description |
|------|-------------|
| `outputs/processed/descriptions_raw.csv` | All extracted phrases (before filtering) |
| `outputs/processed/descriptions.csv` | Filtered descriptions (719 rows) |
| `outputs/reports/task2_rationale.md` | Extraction methodology |

### Task 3: Embedding & Clustering

| File | Description |
|------|-------------|
| `outputs/processed/descriptions_with_clusters.csv` | DBSCAN cluster assignments |
| `outputs/processed/cluster_summary.csv` | Cluster statistics |
| `outputs/figures/umap_clusters_victim.png` | UMAP visualization (victim) |
| `outputs/figures/umap_clusters_shooter.png` | UMAP visualization (shooter) |
| `outputs/reports/task3_doc.md` | Clustering methodology |
| `outputs/reports/dbscan_tuning.md` | Parameter tuning report |

### Task 4: Manual Evaluation & Refinement

| File | Description |
|------|-------------|
| `outputs/reports/task4_manual_eval.md` | Manual evaluation writeup |
| `outputs/processed/cluster_refinement_map.json` | Refinement mapping rules |
| `outputs/processed/descriptions_with_clusters_refined.csv` | Refined cluster labels |

### Task 5: Frequency Analysis

| File | Description |
|------|-------------|
| `outputs/processed/frequency_table_victim.csv` | Raw counts (victim) |
| `outputs/processed/frequency_table_shooter.csv` | Raw counts (shooter) |
| `outputs/processed/proportion_table_victim.csv` | Proportions (victim) |
| `outputs/processed/proportion_table_shooter.csv` | Proportions (shooter) |
| `outputs/figures/task5_heatmap_victim.png` | Heatmap (victim) |
| `outputs/figures/task5_heatmap_shooter.png` | Heatmap (shooter) |
| `outputs/figures/task5_bar_top6_victim.png` | Bar chart (victim) |
| `outputs/figures/task5_bar_top6_shooter.png` | Bar chart (shooter) |
| `outputs/reports/task5_doc.md` | Analysis documentation |

### Task 6: Chi-Square Testing

| File | Description |
|------|-------------|
| `outputs/processed/task6_chi_square_results.csv` | Test results |
| `outputs/reports/task6_results.md` | Hypothesis testing report |
| `outputs/figures/task6_observed_vs_expected_victim_1.png` | Observed vs Expected |
| `outputs/figures/task6_observed_vs_expected_victim_2.png` | Observed vs Expected |
| `outputs/figures/task6_observed_vs_expected_victim_3.png` | Observed vs Expected |
| `outputs/figures/task6_observed_vs_expected_shooter_1.png` | Observed vs Expected |
| `outputs/figures/task6_observed_vs_expected_shooter_2.png` | Observed vs Expected |
| `outputs/figures/task6_observed_vs_expected_shooter_3.png` | Observed vs Expected |

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No module named 'spacy'` | `pip install spacy` |
| `Can't find model 'en_core_web_sm'` | `python -m spacy download en_core_web_sm` |
| `No module named 'sentence_transformers'` | `pip install sentence-transformers` |
| `No module named 'umap'` | `pip install umap-learn` |
| `No module named 'matplotlib'` | `pip install matplotlib` |
| `No module named 'tabulate'` | `pip install tabulate` |
| `No module named 'fastcoref'` | `pip install torch && pip install fastcoref` |

---

## Data Summary

- **100 articles** (25 per outlet: CNN, Fox, NYT, WSJ)
- **719 descriptions** (401 victim, 318 shooter)
- **Key finding**: "Shooter identity labels" cluster shows significant cross-outlet variation (p < 0.001)

---

## License

Educational use only.
