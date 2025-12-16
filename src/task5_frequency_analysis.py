"""
Task 5: Cross-Outlet Frequency Analysis

This module analyzes the distribution of refined cluster labels across news outlets.
It uses the manually refined cluster labels from Task 4.

Input:
    outputs/processed/descriptions_with_clusters_refined.csv

Outputs:
    - outputs/processed/frequency_table_victim.csv
    - outputs/processed/frequency_table_shooter.csv
    - outputs/processed/proportion_table_victim.csv
    - outputs/processed/proportion_table_shooter.csv
    - outputs/figures/task5_heatmap_victim.png
    - outputs/figures/task5_heatmap_shooter.png
    - outputs/figures/task5_bar_top6_victim.png
    - outputs/figures/task5_bar_top6_shooter.png
    - outputs/reports/task5_doc.md

Run with:
    python -m src.task5_frequency_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple


# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

INPUT_FILE = PROCESSED_DIR / "descriptions_with_clusters_refined.csv"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FREQUENCY TABLE FUNCTIONS
# =============================================================================
def load_refined_data() -> pd.DataFrame:
    """Load the refined clustering data from Task 4."""
    df = pd.read_csv(INPUT_FILE)
    return df


def build_frequency_table(df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    """
    Build frequency table for a specific entity type.
    
    Args:
        df: Full DataFrame with all descriptions
        entity_type: 'victim' or 'shooter'
    
    Returns:
        DataFrame with rows=cluster_label_refined, columns=outlets + Total
    """
    # STEP 1: Filter by entity_type FIRST
    entity_df = df[df['entity_type'] == entity_type].copy()
    
    # STEP 2: Create crosstab (each row counted exactly once)
    freq_table = pd.crosstab(
        index=entity_df['cluster_label_refined'],
        columns=entity_df['outlet']
    )
    
    # Ensure column order is consistent
    outlet_order = ['CNN', 'Fox', 'NYT', 'WSJ']
    for col in outlet_order:
        if col not in freq_table.columns:
            freq_table[col] = 0
    freq_table = freq_table[outlet_order]
    
    # STEP 3: Add Total column (row-wise sum)
    freq_table['Total'] = freq_table.sum(axis=1)
    
    # Sort by Total descending
    freq_table = freq_table.sort_values('Total', ascending=False)
    
    return freq_table


def build_proportion_table(freq_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build proportion table from frequency table.
    Normalizes within each outlet column so each column sums to 100.
    
    Args:
        freq_table: Frequency table with outlets as columns
    
    Returns:
        DataFrame with proportions (percentages)
    """
    # Use only outlet columns (exclude Total)
    outlet_cols = ['CNN', 'Fox', 'NYT', 'WSJ']
    freq_outlets = freq_table[outlet_cols]
    
    # Normalize each column to sum to 100%
    prop_table = (freq_outlets / freq_outlets.sum(axis=0) * 100).round(2)
    
    return prop_table


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_heatmap(prop_table: pd.DataFrame, entity_type: str, output_path: Path) -> None:
    """Create heatmap visualization of proportion table."""
    # Sort by row totals (sum across outlets)
    row_totals = prop_table.sum(axis=1)
    sorted_table = prop_table.loc[row_totals.sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(sorted_table) * 0.45)))
    
    # Create heatmap
    im = ax.imshow(sorted_table.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(sorted_table.columns)))
    ax.set_yticks(range(len(sorted_table.index)))
    ax.set_xticklabels(sorted_table.columns, fontsize=11)
    ax.set_yticklabels(sorted_table.index, fontsize=9)
    
    # Add text annotations
    for i in range(len(sorted_table.index)):
        for j in range(len(sorted_table.columns)):
            val = sorted_table.iloc[i, j]
            color = 'white' if val > 18 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                   fontsize=8, color=color)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=10)
    
    # Labels and title
    ax.set_title(f'{entity_type.capitalize()} Cluster Proportions by Outlet', 
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Outlet', fontsize=11)
    ax.set_ylabel('Refined Cluster Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_bar_chart(freq_table: pd.DataFrame, entity_type: str, 
                     output_path: Path, top_n: int = 6) -> None:
    """Create grouped bar chart for top N clusters."""
    # Get top N clusters by Total
    top_clusters = freq_table.nlargest(top_n, 'Total').index.tolist()
    top_table = freq_table.loc[top_clusters, ['CNN', 'Fox', 'NYT', 'WSJ']]
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    x = np.arange(len(top_clusters))
    width = 0.18
    outlets = ['CNN', 'Fox', 'NYT', 'WSJ']
    colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea']
    
    for i, outlet in enumerate(outlets):
        offset = (i - len(outlets)/2 + 0.5) * width
        bars = ax.bar(x + offset, top_table[outlet], width, 
                     label=outlet, color=colors[i], edgecolor='white', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, top_table[outlet]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       str(int(val)), ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Format x-axis labels (truncate long names)
    short_labels = [c[:28] + '...' if len(c) > 28 else c for c in top_clusters]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha='right', fontsize=9)
    
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Top {top_n} {entity_type.capitalize()} Clusters by Outlet', 
                fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# DOCUMENTATION
# =============================================================================
def generate_documentation(df: pd.DataFrame, 
                          freq_tables: Dict[str, pd.DataFrame],
                          prop_tables: Dict[str, pd.DataFrame]) -> str:
    """Generate Task 5 documentation markdown."""
    
    # Calculate statistics for observations
    victim_prop = prop_tables['victim']
    shooter_prop = prop_tables['shooter']
    
    # Find notable differences
    observations = []
    
    # Victim observations
    if 'Victim age & count framing' in victim_prop.index:
        row = victim_prop.loc['Victim age & count framing']
        max_outlet = row.idxmax()
        min_outlet = row.idxmin()
        observations.append(
            f"**Victim age & count framing**: {max_outlet} ({row[max_outlet]:.1f}%) uses this "
            f"more than {min_outlet} ({row[min_outlet]:.1f}%)"
        )
    
    if 'Child victim harm framing' in victim_prop.index:
        row = victim_prop.loc['Child victim harm framing']
        max_outlet = row.idxmax()
        min_outlet = row.idxmin()
        observations.append(
            f"**Child victim harm framing**: Ranges from {row.min():.1f}% ({min_outlet}) "
            f"to {row.max():.1f}% ({max_outlet})"
        )
    
    # Shooter observations
    if 'Shooter identity labels' in shooter_prop.index:
        row = shooter_prop.loc['Shooter identity labels']
        max_outlet = row.idxmax()
        min_outlet = row.idxmin()
        observations.append(
            f"**Shooter identity labels**: {max_outlet} ({row[max_outlet]:.1f}%) vs "
            f"{min_outlet} ({row[min_outlet]:.1f}%) - significant variation"
        )
    
    if 'Alleged/suspected shooter framing' in shooter_prop.index:
        row = shooter_prop.loc['Alleged/suspected shooter framing']
        max_outlet = row.idxmax()
        observations.append(
            f"**Legal hedging (Alleged/suspected)**: {max_outlet} leads with {row[max_outlet]:.1f}%"
        )
    
    observations_text = "\n".join([f"- {obs}" for obs in observations[:4]])
    
    # Top clusters
    top_victim = freq_tables['victim'].nlargest(5, 'Total')[['Total']]
    top_shooter = freq_tables['shooter'].nlargest(5, 'Total')[['Total']]
    
    doc = f'''# Task 5: Cross-Outlet Frequency Analysis

## Overview

This analysis examines how different news outlets (CNN, Fox News, NYT, WSJ) distribute 
their framing choices when describing victims and shooters in mass shooting coverage.

**Note**: This analysis uses the **refined cluster labels from Task 4**, not the original 
DBSCAN cluster assignments. The refinement process involved manual evaluation and 
splitting/merging of clusters to improve semantic coherence.

## Data Source

- **Input**: `outputs/processed/descriptions_with_clusters_refined.csv`
- **Total descriptions**: {len(df)}
- **Victim descriptions**: {len(df[df['entity_type'] == 'victim'])}
- **Shooter descriptions**: {len(df[df['entity_type'] == 'shooter'])}

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
{top_victim.to_markdown()}

### Shooter Clusters (by total count)
{top_shooter.to_markdown()}

## Cross-Outlet Observations

{observations_text}

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
'''
    return doc


# =============================================================================
# VERIFICATION
# =============================================================================
def verify_tables(df: pd.DataFrame, freq_tables: Dict[str, pd.DataFrame]) -> bool:
    """Verify that frequency tables have correct totals."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    for entity_type in ['victim', 'shooter']:
        # Original row count
        original_rows = len(df[df['entity_type'] == entity_type])
        
        # Sum of outlet columns (excluding Total column)
        freq = freq_tables[entity_type]
        outlet_cols = ['CNN', 'Fox', 'NYT', 'WSJ']
        table_sum = freq[outlet_cols].values.sum()
        
        expected = 401 if entity_type == 'victim' else 318
        passed = (original_rows == expected) and (table_sum == expected)
        
        print(f"\n{entity_type.upper()}:")
        print(f"  {entity_type}_original_rows = {original_rows}")
        print(f"  {entity_type}_table_sum_outlets = {table_sum}")
        print(f"  Expected: {expected}")
        print(f"  CHECK: {entity_type}_original_rows == {expected} -> {original_rows == expected}")
        print(f"  CHECK: {entity_type}_table_sum_outlets == {expected} -> {table_sum == expected}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def verify_proportions(prop_tables: Dict[str, pd.DataFrame]) -> None:
    """Verify that proportion table columns sum to ~100."""
    print("\n" + "-" * 60)
    print("PROPORTION TABLE COLUMN SUMS (should be ~100)")
    print("-" * 60)
    
    for entity_type in ['victim', 'shooter']:
        prop = prop_tables[entity_type]
        col_sums = prop.sum()
        print(f"\n{entity_type.upper()}:")
        for col, val in col_sums.items():
            status = "✓" if 99.9 <= val <= 100.1 else "✗"
            print(f"  {col}: {val:.2f}% {status}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run Task 5: Cross-Outlet Frequency Analysis."""
    print("=" * 60)
    print("Task 5: Cross-Outlet Frequency Analysis")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading: {INPUT_FILE}")
    df = load_refined_data()
    print(f"Total rows: {len(df)}")
    print(f"Entity types: {df['entity_type'].value_counts().to_dict()}")
    
    # Build frequency tables
    print("\n" + "-" * 60)
    print("Building Frequency Tables")
    print("-" * 60)
    
    freq_tables = {}
    for entity_type in ['victim', 'shooter']:
        freq_tables[entity_type] = build_frequency_table(df, entity_type)
        output_path = PROCESSED_DIR / f"frequency_table_{entity_type}.csv"
        freq_tables[entity_type].to_csv(output_path)
        print(f"Saved: {output_path}")
    
    # Build proportion tables
    print("\n" + "-" * 60)
    print("Building Proportion Tables")
    print("-" * 60)
    
    prop_tables = {}
    for entity_type in ['victim', 'shooter']:
        prop_tables[entity_type] = build_proportion_table(freq_tables[entity_type])
        output_path = PROCESSED_DIR / f"proportion_table_{entity_type}.csv"
        prop_tables[entity_type].to_csv(output_path)
        print(f"Saved: {output_path}")
    
    # Verification
    verify_tables(df, freq_tables)
    verify_proportions(prop_tables)
    
    # Create visualizations
    print("\n" + "-" * 60)
    print("Creating Visualizations")
    print("-" * 60)
    
    for entity_type in ['victim', 'shooter']:
        # Heatmap
        heatmap_path = FIGURES_DIR / f"task5_heatmap_{entity_type}.png"
        create_heatmap(prop_tables[entity_type], entity_type, heatmap_path)
        print(f"Saved: {heatmap_path}")
        
        # Bar chart
        bar_path = FIGURES_DIR / f"task5_bar_top6_{entity_type}.png"
        create_bar_chart(freq_tables[entity_type], entity_type, bar_path, top_n=6)
        print(f"Saved: {bar_path}")
    
    # Generate documentation
    print("\n" + "-" * 60)
    print("Generating Documentation")
    print("-" * 60)
    
    doc_content = generate_documentation(df, freq_tables, prop_tables)
    doc_path = REPORTS_DIR / "task5_doc.md"
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    print(f"Saved: {doc_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TASK 5 COMPLETE")
    print("=" * 60)
    
    print("\nFiles created:")
    print("  Frequency tables:")
    print("    - outputs/processed/frequency_table_victim.csv")
    print("    - outputs/processed/frequency_table_shooter.csv")
    print("  Proportion tables:")
    print("    - outputs/processed/proportion_table_victim.csv")
    print("    - outputs/processed/proportion_table_shooter.csv")
    print("  Figures:")
    print("    - outputs/figures/task5_heatmap_victim.png")
    print("    - outputs/figures/task5_heatmap_shooter.png")
    print("    - outputs/figures/task5_bar_top6_victim.png")
    print("    - outputs/figures/task5_bar_top6_shooter.png")
    print("  Documentation:")
    print("    - outputs/reports/task5_doc.md")


if __name__ == "__main__":
    main()

