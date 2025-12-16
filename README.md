# HW5 NLP Pipeline — Media Framing Analysis (Victims vs Shooters)

This project analyzes how different news outlets (CNN, Fox News, NYT, WSJ) frame **victims** and **shooters** in mass shooting coverage using NLP techniques.

## Pipeline Overview

1. **Task 1**: Load articles → Coreference resolution → Extract victim/shooter contexts
2. **Task 2**: Extract descriptive phrases using dependency parsing
3. **Task 3**: Embed descriptions with SBERT → Cluster with DBSCAN
4. **Task 4**: Manual cluster evaluation and refinement
5. **Task 5**: Cross-outlet frequency and proportion analysis
6. **Task 6**: Chi-squared hypothesis testing for outlet differences

---

## Project Structure

```
hw5-NLP/
├── data_100/                          # Raw article data (DO NOT MODIFY)
│   ├── cnn_five_para/
│   ├── FOX_five_para/
│   ├── NYT_five_para/
│   └── WSJ_five_para/
├── outputs/
│   ├── processed/                     # CSV, JSONL, JSON outputs
│   ├── figures/                       # PNG visualizations
│   └── reports/                       # Markdown documentation
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration and paths
│   ├── utils_io.py                    # I/O utilities
│   ├── load_articles.py               # Task 1a: Load raw articles
│   ├── coref_contexts.py              # Task 1b: Coreference + context extraction
│   ├── extract_descriptions.py        # Task 2: Description phrase extraction
│   ├── embed_cluster.py               # Task 3: Embedding + DBSCAN clustering
│   ├── manual_eval_helpers.py         # Task 4: Cluster evaluation + refinement
│   ├── task5_frequency_analysis.py    # Task 5: Frequency/proportion tables
│   ├── task6_chi_square.py            # Task 6: Chi-squared hypothesis tests
│   ├── freq_tables.py                 # (Legacy) frequency tables
│   ├── stats_tests.py                 # (Legacy) statistical tests
│   └── run_all.py                     # Run full pipeline (Tasks 1-3)
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Option 1: Using venv (recommended for Mac)

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n hw5nlp python=3.10 -y
conda activate hw5nlp

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Verify Installation

```bash
python -c "import spacy; import fastcoref; import sentence_transformers; import umap; import matplotlib; import scipy; print('All imports successful!')"
```

---

## Running the Pipeline

Execute each task in order from the project root directory.

### Task 1: Load Articles & Coreference Resolution

```bash
# Load raw articles into DataFrame
python -m src.load_articles

# Run coreference resolution and extract victim/shooter contexts
python -m src.coref_contexts
```

### Task 2: Extract Descriptive Phrases

```bash
python -m src.extract_descriptions
```

### Task 3: Embed & Cluster Descriptions

```bash
python -m src.embed_cluster
```

### Task 4: Manual Cluster Evaluation & Refinement

```bash
python -m src.manual_eval_helpers
```

### Task 5: Frequency & Proportion Analysis

```bash
python -m src.task5_frequency_analysis
```

### Task 6: Chi-Squared Hypothesis Testing

```bash
python -m src.task6_chi_square
```

---

## Output Files

### Task 1 Outputs

| File | Description |
|------|-------------|
| `outputs/processed/articles_master.csv` | All articles with metadata |
| `outputs/processed/contexts_victims.jsonl` | Victim-related sentences (coref-resolved) |
| `outputs/processed/contexts_shooters.jsonl` | Shooter-related sentences (coref-resolved) |
| `outputs/reports/task1_doc.md` | Methodology documentation |

### Task 2 Outputs

| File | Description |
|------|-------------|
| `outputs/processed/descriptions_raw.csv` | All extracted phrases (before filtering) |
| `outputs/processed/descriptions.csv` | Filtered descriptive phrases |
| `outputs/reports/task2_rationale.md` | Extraction methodology |

### Task 3 Outputs

| File | Description |
|------|-------------|
| `outputs/processed/descriptions_with_clusters.csv` | Descriptions with cluster assignments |
| `outputs/processed/cluster_summary.csv` | Cluster statistics and examples |
| `outputs/figures/umap_clusters_victim.png` | UMAP visualization (victim) |
| `outputs/figures/umap_clusters_shooter.png` | UMAP visualization (shooter) |
| `outputs/reports/dbscan_tuning.md` | DBSCAN parameter tuning report |
| `outputs/reports/task3_doc.md` | Clustering methodology |

### Task 4 Outputs

| File | Description |
|------|-------------|
| `outputs/reports/task4_manual_eval.md` | Manual cluster evaluation report |
| `outputs/processed/cluster_refinement_map.json` | Cluster refinement mapping |
| `outputs/processed/descriptions_with_clusters_refined.csv` | Refined cluster labels |

### Task 5 Outputs

| File | Description |
|------|-------------|
| `outputs/processed/frequency_table_victim.csv` | Raw counts by outlet (victim) |
| `outputs/processed/frequency_table_shooter.csv` | Raw counts by outlet (shooter) |
| `outputs/processed/proportion_table_victim.csv` | Column-normalized % (victim) |
| `outputs/processed/proportion_table_shooter.csv` | Column-normalized % (shooter) |
| `outputs/figures/task5_heatmap_victim.png` | Proportion heatmap (victim) |
| `outputs/figures/task5_heatmap_shooter.png` | Proportion heatmap (shooter) |
| `outputs/figures/task5_bar_top6_victim.png` | Top 6 clusters bar chart (victim) |
| `outputs/figures/task5_bar_top6_shooter.png` | Top 6 clusters bar chart (shooter) |
| `outputs/reports/task5_doc.md` | Frequency analysis documentation |

### Task 6 Outputs

| File | Description |
|------|-------------|
| `outputs/processed/task6_chi_square_results.csv` | Chi-square test results |
| `outputs/reports/task6_results.md` | Hypothesis testing report |
| `outputs/figures/task6_observed_vs_expected_victim_1.png` | Observed vs Expected (victim cluster 1) |
| `outputs/figures/task6_observed_vs_expected_victim_2.png` | Observed vs Expected (victim cluster 2) |
| `outputs/figures/task6_observed_vs_expected_victim_3.png` | Observed vs Expected (victim cluster 3) |
| `outputs/figures/task6_observed_vs_expected_shooter_1.png` | Observed vs Expected (shooter cluster 1) |
| `outputs/figures/task6_observed_vs_expected_shooter_2.png` | Observed vs Expected (shooter cluster 2) |
| `outputs/figures/task6_observed_vs_expected_shooter_3.png` | Observed vs Expected (shooter cluster 3) |

---

## Troubleshooting

### "No module named spacy"
```bash
# Ensure you're in the correct environment
which python
pip install spacy
```

### spaCy model missing
```bash
python -m spacy download en_core_web_sm
```

### "No module named tabulate"
```bash
pip install tabulate
```

### "No module named matplotlib"
```bash
pip install matplotlib
```

### fastcoref installation issues
```bash
# Try installing with pip
pip install fastcoref

# If issues persist, ensure torch is installed first
pip install torch
pip install fastcoref
```

### UMAP installation issues (Mac M1/M2)
```bash
pip install umap-learn
```

---

## Data Summary

- **4 outlets**: CNN, Fox News, NYT, WSJ
- **~100 articles** per outlet (5 paragraphs each)
- **719 total descriptions** extracted and clustered
- **401 victim descriptions**, **318 shooter descriptions**

---

## Key Findings

- **Shooter identity labels** cluster shows statistically significant variation across outlets (p = 0.000011)
- CNN overuses generic shooter labels; Fox underuses them
- Victim framing is more consistent across outlets

---

## License

Educational use only. Part of NLP coursework.
