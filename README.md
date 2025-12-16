# Comparative Analysis of Gun Violence Coverage Across News Outlets

This project analyzes how four major U.S. news outlets (CNN, Fox News, NYT, WSJ) frame **victims** and **shooters** in mass shooting coverage. Using 100 articles, we extract victim/shooter contexts via coreference resolution, extract descriptive phrases using dependency parsing, cluster descriptions with SBERT embeddings and DBSCAN, apply manual refinement, compute cross-outlet frequency/proportion tables, and run chi-squared hypothesis tests to identify statistically significant framing differences.

---

## Project Structure

```
hw5-NLP/
├── data_100/                              # Raw articles (DO NOT MODIFY)
│   ├── cnn_five_para/                     # 25 CNN articles
│   ├── FOX_five_para/                     # 25 Fox News articles
│   ├── NYT_five_para/                     # 25 NYT articles
│   └── WSJ_five_para/                     # 25 WSJ articles
├── outputs/
│   ├── processed/                         # CSV, JSONL, JSON data files
│   ├── figures/                           # PNG visualizations
│   └── reports/                           # Markdown documentation
├── src/
│   ├── __init__.py
│   ├── config.py                          # Configuration and paths
│   ├── utils_io.py                        # I/O utilities
│   ├── load_articles.py                   # Task 1a: Load articles
│   ├── coref_contexts.py                  # Task 1b: Coreference resolution
│   ├── extract_descriptions.py            # Task 2: Extract descriptions
│   ├── embed_cluster.py                   # Task 3: Embedding + clustering
│   ├── manual_eval_helpers.py             # Task 4: Manual evaluation + refinement
│   ├── task5_frequency_analysis.py        # Task 5: Frequency/proportion tables
│   └── task6_chi_square.py                # Task 6: Chi-squared hypothesis tests
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Option 1: venv (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Option 2: conda

```bash
conda create -n hw5nlp python=3.10 -y
conda activate hw5nlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Sanity Check

```bash
python -c "import spacy, pandas, sentence_transformers, sklearn, scipy, matplotlib, tabulate; print('All imports OK')"
```

---

## Running the Pipeline

Run each task in order from the project root directory:

```bash
# Task 1a: Load articles into DataFrame
python -m src.load_articles

# Task 1b: Coreference resolution + context extraction
python -m src.coref_contexts

# Task 2: Extract descriptive phrases
python -m src.extract_descriptions

# Task 3: SBERT embedding + DBSCAN clustering
python -m src.embed_cluster

# Task 4: Manual cluster evaluation + refinement
python -m src.manual_eval_helpers

# Task 5: Frequency/proportion analysis + visualizations
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
| `outputs/processed/cluster_summary.csv` | Cluster statistics and examples |
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
| `outputs/processed/frequency_table_victim.csv` | Raw counts by outlet (victim) |
| `outputs/processed/frequency_table_shooter.csv` | Raw counts by outlet (shooter) |
| `outputs/processed/proportion_table_victim.csv` | Column-normalized % (victim) |
| `outputs/processed/proportion_table_shooter.csv` | Column-normalized % (shooter) |
| `outputs/figures/task5_heatmap_victim.png` | Proportion heatmap (victim) |
| `outputs/figures/task5_heatmap_shooter.png` | Proportion heatmap (shooter) |
| `outputs/figures/task5_bar_top6_victim.png` | Top 6 clusters bar chart (victim) |
| `outputs/figures/task5_bar_top6_shooter.png` | Top 6 clusters bar chart (shooter) |
| `outputs/reports/task5_doc.md` | Analysis documentation |

### Task 6: Chi-Square Testing

| File | Description |
|------|-------------|
| `outputs/processed/task6_chi_square_results.csv` | Chi-square test results (6 tests) |
| `outputs/reports/task6_results.md` | Hypothesis testing report |
| `outputs/figures/task6_observed_vs_expected_victim_1.png` | Observed vs Expected plot |
| `outputs/figures/task6_observed_vs_expected_victim_2.png` | Observed vs Expected plot |
| `outputs/figures/task6_observed_vs_expected_victim_3.png` | Observed vs Expected plot |
| `outputs/figures/task6_observed_vs_expected_shooter_1.png` | Observed vs Expected plot |
| `outputs/figures/task6_observed_vs_expected_shooter_2.png` | Observed vs Expected plot |
| `outputs/figures/task6_observed_vs_expected_shooter_3.png` | Observed vs Expected plot |

---

## Verification Checks

After running the pipeline, verify correctness:

```bash
# Frequency table sums should match original counts (victim=401, shooter=318)
python -c "
import pandas as pd
v = pd.read_csv('outputs/processed/frequency_table_victim.csv', index_col=0)
s = pd.read_csv('outputs/processed/frequency_table_shooter.csv', index_col=0)
print(f'Victim sum: {v[[\"CNN\",\"Fox\",\"NYT\",\"WSJ\"]].values.sum()}')  # Expected: 401
print(f'Shooter sum: {s[[\"CNN\",\"Fox\",\"NYT\",\"WSJ\"]].values.sum()}')  # Expected: 318
"

# Proportion columns should sum to ~100%
python -c "
import pandas as pd
p = pd.read_csv('outputs/processed/proportion_table_victim.csv', index_col=0)
print('Column sums:', p.sum().round(1).to_dict())  # Each ~100
"
```

---

## Data Summary

- **100 articles** (25 per outlet: CNN, Fox, NYT, WSJ)
- **719 descriptions** extracted and clustered
- **401 victim descriptions**, **318 shooter descriptions**
- **Key finding**: "Shooter identity labels" shows significant cross-outlet variation (χ² = 25.72, p < 0.001)

---

## License

For academic/course use only.
