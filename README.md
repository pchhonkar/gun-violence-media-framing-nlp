# HW5 NLP Pipeline — Media Framing Analysis (Victims vs Shooters)

This project analyzes how different news outlets (CNN, Fox News, NYT, WSJ) frame **victims** and **shooters** in mass shooting coverage using NLP techniques including coreference resolution, dependency parsing, sentence embeddings, and statistical hypothesis testing.

---

## 📋 Note for Graders

- **Python scripts in `src/` are the authoritative implementation** for all tasks
- All pipeline steps can be run via command line using `python -m src.<module>`
- The complete pipeline can be executed with a single command: `python -m src.run_all`
- Any Jupyter notebooks (if present) are optional and not required for grading

---

## Pipeline Overview

| Task | Description | Key Output |
|------|-------------|------------|
| **Task 1** | Load articles + Coreference resolution | `contexts_victims.jsonl`, `contexts_shooters.jsonl` |
| **Task 2** | Extract descriptive phrases | `descriptions.csv` |
| **Task 3** | Embed with SBERT + Cluster with DBSCAN | `descriptions_with_clusters.csv` |
| **Task 4** | Manual cluster evaluation & refinement | `descriptions_with_clusters_refined.csv` |
| **Task 5** | Cross-outlet frequency analysis | Frequency/proportion tables + heatmaps |
| **Task 6** | Chi-squared hypothesis testing | `task6_chi_square_results.csv` |

---

## Project Structure

```
hw5-NLP/
├── data_100/                              # Raw article data (DO NOT MODIFY)
│   ├── cnn_five_para/                     # 25 CNN articles
│   ├── FOX_five_para/                     # 25 Fox News articles
│   ├── NYT_five_para/                     # 25 NYT articles
│   └── WSJ_five_para/                     # 25 WSJ articles
├── outputs/
│   ├── processed/                         # CSV, JSONL, JSON outputs
│   ├── figures/                           # PNG visualizations
│   └── reports/                           # Markdown documentation
├── src/
│   ├── __init__.py
│   ├── config.py                          # Configuration and paths
│   ├── utils_io.py                        # I/O utilities
│   ├── load_articles.py                   # Task 1a: Load raw articles
│   ├── coref_contexts.py                  # Task 1b: Coreference + context extraction
│   ├── extract_descriptions.py            # Task 2: Description phrase extraction
│   ├── embed_cluster.py                   # Task 3: Embedding + DBSCAN clustering
│   ├── manual_eval_helpers.py             # Task 4: Cluster evaluation + refinement
│   ├── task5_frequency_analysis.py        # Task 5: Frequency/proportion tables
│   ├── task6_chi_square.py                # Task 6: Chi-squared hypothesis tests
│   └── run_all.py                         # Run complete pipeline
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
python -c "import spacy; import pandas; import numpy; import sklearn; import scipy; import matplotlib; import tabulate; print('All imports successful!')"
```

---

## Running the Pipeline

### Complete Pipeline (Recommended)

```bash
# Run all tasks in order (skips steps if outputs exist)
python -m src.run_all

# Force re-run all steps
python -m src.run_all --force

# Skip Task 4 (manual refinement)
python -m src.run_all --skip-task4
```

### Individual Steps

```bash
# Task 1a: Load articles
python -m src.load_articles

# Task 1b: Coreference resolution + context extraction
python -m src.coref_contexts

# Task 2: Extract descriptive phrases
python -m src.extract_descriptions

# Task 3: Embedding + DBSCAN clustering
python -m src.embed_cluster

# Task 4: Manual evaluation + refinement
python -m src.manual_eval_helpers

# Task 5: Frequency/proportion tables
python -m src.task5_frequency_analysis

# Task 6: Chi-squared hypothesis testing
python -m src.task6_chi_square
```

---

## Output Files

### Task 1: Article Loading & Coreference

| File | Description |
|------|-------------|
| `outputs/processed/articles_master.csv` | All 100 articles with metadata |
| `outputs/processed/contexts_victims.jsonl` | Victim-related sentences (coref-resolved) |
| `outputs/processed/contexts_shooters.jsonl` | Shooter-related sentences (coref-resolved) |
| `outputs/reports/task1_doc.md` | Methodology documentation |

### Task 2: Description Extraction

| File | Description |
|------|-------------|
| `outputs/processed/descriptions_raw.csv` | All extracted phrases (before filtering) |
| `outputs/processed/descriptions.csv` | Filtered descriptive phrases (719 rows) |
| `outputs/reports/task2_rationale.md` | Extraction methodology |

### Task 3: Embedding & Clustering

| File | Description |
|------|-------------|
| `outputs/processed/descriptions_with_clusters.csv` | Descriptions with DBSCAN cluster assignments |
| `outputs/processed/cluster_summary.csv` | Cluster statistics and examples |
| `outputs/figures/umap_clusters_victim.png` | UMAP visualization (victim) |
| `outputs/figures/umap_clusters_shooter.png` | UMAP visualization (shooter) |
| `outputs/reports/dbscan_tuning.md` | DBSCAN parameter tuning report |
| `outputs/reports/task3_doc.md` | Clustering methodology |

### Task 4: Manual Evaluation & Refinement

| File | Description |
|------|-------------|
| `outputs/reports/task4_manual_eval.md` | Manual cluster evaluation report |
| `outputs/processed/cluster_refinement_map.json` | Cluster refinement mapping rules |
| `outputs/processed/descriptions_with_clusters_refined.csv` | Refined cluster labels (719 rows) |

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
| `outputs/reports/task5_doc.md` | Frequency analysis documentation |

### Task 6: Chi-Square Testing

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

### Pipeline Log

| File | Description |
|------|-------------|
| `outputs/reports/run_log.txt` | Complete pipeline execution log |

---

## Quick Verification Commands

Run these commands to verify the pipeline outputs are correct:

```bash
# Verify article count (should be 100)
python -c "import pandas as pd; df=pd.read_csv('outputs/processed/articles_master.csv'); print(f'Articles: {len(df)}')"

# Verify description counts (should be 719 total: 401 victim + 318 shooter)
python -c "import pandas as pd; df=pd.read_csv('outputs/processed/descriptions_with_clusters_refined.csv'); print(f'Total: {len(df)}, Victim: {len(df[df.entity_type==\"victim\"])}, Shooter: {len(df[df.entity_type==\"shooter\"])}')"

# Verify Task 5 frequency table sums (victim=401, shooter=318)
python -c "import pandas as pd; v=pd.read_csv('outputs/processed/frequency_table_victim.csv',index_col=0); s=pd.read_csv('outputs/processed/frequency_table_shooter.csv',index_col=0); print(f'Victim sum: {v[[\"CNN\",\"Fox\",\"NYT\",\"WSJ\"]].values.sum()}, Shooter sum: {s[[\"CNN\",\"Fox\",\"NYT\",\"WSJ\"]].values.sum()}')"

# Verify proportion tables sum to ~100% per column
python -c "import pandas as pd; p=pd.read_csv('outputs/processed/proportion_table_victim.csv',index_col=0); print('Victim proportion column sums:'); print(p.sum())"
```

**Expected output:**
```
Articles: 100
Total: 719, Victim: 401, Shooter: 318
Victim sum: 401, Shooter sum: 318
Victim proportion column sums:
CNN     99.99
Fox     99.99
NYT    100.00
WSJ     99.99
```

---

## Key Findings

- **Shooter identity labels** cluster shows statistically significant variation across outlets (χ² = 25.72, p = 0.000011)
- **CNN overuses** generic shooter labels ("the gunman", "the shooter")
- **Fox underuses** generic labels (uses more specific framing like "alleged shooter")
- Victim framing is more consistent across outlets (no significant differences at α = 0.05)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named spacy` | `pip install spacy` |
| spaCy model missing | `python -m spacy download en_core_web_sm` |
| `No module named tabulate` | `pip install tabulate` |
| `No module named matplotlib` | `pip install matplotlib` |
| fastcoref issues | `pip install torch && pip install fastcoref` |

---

## Data Summary

- **4 outlets**: CNN, Fox News, NYT, WSJ
- **100 articles** total (25 per outlet, 5 paragraphs each)
- **719 descriptions** extracted and clustered
- **401 victim descriptions**, **318 shooter descriptions**

---

## License

Educational use only. Part of NLP coursework.
