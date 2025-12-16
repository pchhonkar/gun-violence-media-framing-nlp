# Task 5: Cross-Outlet Frequency Analysis

## Overview

This analysis examines how different news outlets (CNN, Fox News, NYT, WSJ) distribute 
their framing choices when describing victims and shooters in mass shooting coverage.

**Note**: This analysis uses the **refined cluster labels from Task 4**, not the original 
DBSCAN cluster assignments. The refinement process involved manual evaluation and 
splitting/merging of clusters to improve semantic coherence.

## Data Source

- **Input**: `outputs/processed/descriptions_with_clusters_refined.csv`
- **Total descriptions**: 609
- **Victim descriptions**: 361
- **Shooter descriptions**: 248

## Methodology

### Frequency Tables vs. Proportion Tables

| Aspect | Frequency Table | Proportion Table |
|--------|-----------------|------------------|
| **Values** | Raw counts | Percentages (0-100) |
| **Purpose** | Show absolute usage | Enable cross-outlet comparison |
| **Interpretation** | "How many times" | "What fraction of coverage" |

### Normalization Method

Proportion tables are **column-normalized**:
- Each outlet column sums to 100%
- Formula: `proportion = (cluster_count / outlet_total) × 100`

This normalization allows fair comparison across outlets that may have different 
total coverage volumes. It answers: "Of all victim/shooter descriptions from Outlet X, 
what percentage fall into each cluster?"

## Top Clusters

### Victim Clusters (by total count)
| refined_cluster_label      |   Total |
|:---------------------------|--------:|
| Victim age & count framing |     116 |
| Noise/Unclustered          |      59 |
| Action descriptions        |      55 |
| Harm severity              |      50 |
| Child victim harm framing  |      28 |

### Shooter Clusters (by total count)
| refined_cluster_label             |   Total |
|:----------------------------------|--------:|
| Shooter identity labels           |      87 |
| Noise/Unclustered                 |      59 |
| Harm severity                     |      29 |
| Active threat framing             |      24 |
| Alleged/suspected shooter framing |      17 |

## Cross-Outlet Observations

- **Victim age & count framing**: WSJ (44.1%) uses this more than Fox (28.3%)
- **Child victim harm framing**: Ranges from 2.9% (WSJ) to 10.3% (CNN)
- **Shooter identity labels**: CNN (50.0%) vs Fox (14.7%) - significant variation
- **Legal hedging (Alleged/suspected)**: Fox leads with 17.3%

## Files Generated

| File | Description |
|------|-------------|
| `frequency_table_victim.csv` | Raw counts by cluster × outlet (victim) |
| `frequency_table_shooter.csv` | Raw counts by cluster × outlet (shooter) |
| `proportion_table_victim.csv` | Column-normalized percentages (victim) |
| `proportion_table_shooter.csv` | Column-normalized percentages (shooter) |
| `task5_heatmap_victim.png` | Proportion heatmap (victim) |
| `task5_heatmap_shooter.png` | Proportion heatmap (shooter) |
| `task5_bar_top6_victim.png` | Top 6 clusters bar chart (victim) |
| `task5_bar_top6_shooter.png` | Top 6 clusters bar chart (shooter) |

## Limitations

1. Sample sizes vary by outlet
2. Some clusters are small (< 10 phrases)
3. Refined labels are post-hoc manual assignments from Task 4
4. Original DBSCAN clustering may have inherent biases
