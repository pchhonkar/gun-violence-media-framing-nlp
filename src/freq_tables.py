"""
Frequency table generation module.

Creates frequency and proportion tables for cluster distributions by outlet.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from . import utils_io


def compute_frequency_table(
    df: pd.DataFrame,
    entity_type: str
) -> pd.DataFrame:
    """
    Compute frequency table (raw counts) by outlet x cluster_label.
    
    Args:
        df: Descriptions DataFrame with cluster assignments
        entity_type: "victim" or "shooter"
        
    Returns:
        Frequency table DataFrame (outlets as rows, clusters as columns)
    """
    # Filter to entity type
    filtered = df[df['entity_type'] == entity_type]
    
    # Create cross-tabulation
    freq_table = pd.crosstab(
        filtered['outlet'],
        filtered['cluster_label'],
        margins=True,
        margins_name='Total'
    )
    
    return freq_table


def compute_proportion_table(
    freq_table: pd.DataFrame,
    normalize_by: str = "row"
) -> pd.DataFrame:
    """
    Compute proportion table from frequency table.
    
    Args:
        freq_table: Frequency table DataFrame
        normalize_by: "row" (within outlet) or "column" (within cluster)
        
    Returns:
        Proportion table as percentages
    """
    # Remove margins for calculation
    if 'Total' in freq_table.index:
        data_table = freq_table.drop('Total', axis=0)
    else:
        data_table = freq_table.copy()
    
    if 'Total' in data_table.columns:
        data_table = data_table.drop('Total', axis=1)
    
    if normalize_by == "row":
        # Normalize by row (within each outlet)
        prop_table = data_table.div(data_table.sum(axis=1), axis=0) * 100
    else:
        # Normalize by column (within each cluster)
        prop_table = data_table.div(data_table.sum(axis=0), axis=1) * 100
    
    # Round to 1 decimal place
    prop_table = prop_table.round(1)
    
    return prop_table


def get_top_clusters(
    df: pd.DataFrame,
    entity_type: str,
    n: int = 15
) -> List[str]:
    """
    Get top N clusters by overall frequency.
    
    Args:
        df: Descriptions DataFrame
        entity_type: "victim" or "shooter"
        n: Number of top clusters
        
    Returns:
        List of top cluster labels
    """
    filtered = df[df['entity_type'] == entity_type]
    
    # Count by cluster_label
    cluster_counts = filtered['cluster_label'].value_counts()
    
    # Get top N (excluding noise if present)
    top_clusters = cluster_counts.head(n).index.tolist()
    
    return top_clusters


def plot_heatmap(
    prop_table: pd.DataFrame,
    entity_type: str,
    output_path: Path,
    top_n: int = 15
) -> None:
    """
    Create heatmap of proportions.
    
    Args:
        prop_table: Proportion table
        entity_type: "victim" or "shooter"
        output_path: Path to save figure
        top_n: Number of top clusters to show
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Select top N columns by total frequency
    col_sums = prop_table.sum(axis=0).sort_values(ascending=False)
    top_cols = col_sums.head(top_n).index.tolist()
    
    plot_data = prop_table[top_cols]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    im = ax.imshow(plot_data.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_yticks(range(len(plot_data.index)))
    
    # Set labels with rotation for x-axis
    ax.set_xticklabels(plot_data.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(plot_data.index, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=10)
    
    # Add text annotations
    for i in range(len(plot_data.index)):
        for j in range(len(plot_data.columns)):
            value = plot_data.iloc[i, j]
            if value > 0:
                text_color = 'white' if value > 20 else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                       fontsize=8, color=text_color)
    
    ax.set_title(f'{entity_type.capitalize()} Cluster Distribution by Outlet (Top {top_n})', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster Label', fontsize=10)
    ax.set_ylabel('Outlet', fontsize=10)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def plot_bar_top_clusters(
    df: pd.DataFrame,
    entity_type: str,
    output_path: Path,
    top_n: int = 10
) -> None:
    """
    Create grouped bar chart of top clusters by outlet.
    
    Args:
        df: Descriptions DataFrame
        entity_type: "victim" or "shooter"
        output_path: Path to save figure
        top_n: Number of top clusters to show
    """
    import matplotlib.pyplot as plt
    
    # Filter to entity type
    filtered = df[df['entity_type'] == entity_type]
    
    # Get top clusters
    top_clusters = get_top_clusters(df, entity_type, top_n)
    
    # Filter to top clusters
    filtered_top = filtered[filtered['cluster_label'].isin(top_clusters)]
    
    # Create cross-tabulation
    counts = pd.crosstab(filtered_top['cluster_label'], filtered_top['outlet'])
    
    # Reorder by total frequency
    counts['_total'] = counts.sum(axis=1)
    counts = counts.sort_values('_total', ascending=True)
    counts = counts.drop('_total', axis=1)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    outlets = counts.columns.tolist()
    x = np.arange(len(counts))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # CNN, Fox, NYT, WSJ
    
    for i, outlet in enumerate(outlets):
        offset = (i - len(outlets)/2 + 0.5) * width
        bars = ax.barh(x + offset, counts[outlet], width, label=outlet, color=colors[i % len(colors)])
    
    ax.set_yticks(x)
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel('Count', fontsize=10)
    ax.set_ylabel('Cluster Label', fontsize=10)
    ax.set_title(f'Top {top_n} {entity_type.capitalize()} Clusters by Outlet', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart to: {output_path}")


def create_contingency_table(
    df: pd.DataFrame,
    entity_type: str
) -> pd.DataFrame:
    """
    Create contingency table for statistical testing (clusters x outlets).
    
    Args:
        df: Descriptions DataFrame
        entity_type: "victim" or "shooter"
        
    Returns:
        Contingency table (cluster_id rows, outlet columns)
    """
    filtered = df[df['entity_type'] == entity_type]
    
    contingency = pd.crosstab(
        filtered['cluster_id'],
        filtered['outlet']
    )
    
    return contingency


def format_table_for_report(
    df: pd.DataFrame,
    title: str,
    format: str = "markdown"
) -> str:
    """
    Format a table for inclusion in a report.
    
    Args:
        df: DataFrame to format
        title: Table title
        format: Output format ('markdown', 'latex', 'html')
        
    Returns:
        Formatted table string
    """
    if format == "markdown":
        return f"## {title}\n\n{df.to_markdown()}\n"
    elif format == "latex":
        return f"% {title}\n{df.to_latex()}\n"
    elif format == "html":
        return f"<h2>{title}</h2>\n{df.to_html()}\n"
    else:
        return f"{title}\n{df.to_string()}\n"


def process_entity_type(
    df: pd.DataFrame,
    entity_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process frequency and proportion tables for one entity type.
    
    Args:
        df: Descriptions DataFrame
        entity_type: "victim" or "shooter"
        
    Returns:
        Tuple of (frequency_table, proportion_table)
    """
    print(f"\nProcessing {entity_type} tables...")
    
    # Compute frequency table
    freq_table = compute_frequency_table(df, entity_type)
    print(f"  Frequency table shape: {freq_table.shape}")
    
    # Compute proportion table (row-wise normalization)
    prop_table = compute_proportion_table(freq_table, normalize_by="row")
    print(f"  Proportion table shape: {prop_table.shape}")
    
    return freq_table, prop_table


def main():
    """
    Main function to generate frequency tables and plots.
    Run with: python -m src.freq_tables
    """
    print("=" * 60)
    print("Frequency Table Generation")
    print("=" * 60)
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    
    # Check input file exists
    input_path = config.PROCESSED_DIR / "descriptions_with_clusters.csv"
    if not input_path.exists():
        print(f"Error: Clustered descriptions file not found at {input_path}")
        print("Please run 'python -m src.embed_cluster' first.")
        return
    
    # Load data
    print(f"\nLoading data from {input_path}...")
    df = utils_io.load_dataframe(input_path)
    print(f"Loaded {len(df)} descriptions")
    
    # Process each entity type
    for entity_type in ['victim', 'shooter']:
        freq_table, prop_table = process_entity_type(df, entity_type)
        
        # Save frequency table
        freq_path = config.PROCESSED_DIR / f"freq_table_{entity_type}.csv"
        freq_table.to_csv(freq_path)
        print(f"  Saved frequency table to: {freq_path}")
        
        # Save proportion table
        prop_path = config.PROCESSED_DIR / f"prop_table_{entity_type}.csv"
        prop_table.to_csv(prop_path)
        print(f"  Saved proportion table to: {prop_path}")
        
        # Create heatmap
        heatmap_path = config.FIGURES_DIR / f"heatmap_{entity_type}.png"
        plot_heatmap(prop_table, entity_type, heatmap_path, top_n=15)
        
        # Create bar chart
        bar_path = config.FIGURES_DIR / f"bar_top_clusters_{entity_type}.png"
        plot_bar_top_clusters(df, entity_type, bar_path, top_n=10)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for entity_type in ['victim', 'shooter']:
        filtered = df[df['entity_type'] == entity_type]
        
        print(f"\n{entity_type.upper()}:")
        print(f"  Total descriptions: {len(filtered)}")
        print(f"  Unique clusters: {filtered['cluster_id'].nunique()}")
        
        print("\n  By outlet:")
        outlet_counts = filtered['outlet'].value_counts()
        for outlet, count in outlet_counts.items():
            pct = count / len(filtered) * 100
            print(f"    {outlet}: {count} ({pct:.1f}%)")
        
        print("\n  Top 5 clusters:")
        cluster_counts = filtered['cluster_label'].value_counts().head(5)
        for cluster, count in cluster_counts.items():
            pct = count / len(filtered) * 100
            print(f"    {cluster}: {count} ({pct:.1f}%)")
    
    # Print sample tables
    print("\n" + "=" * 60)
    print("SAMPLE TABLES")
    print("=" * 60)
    
    print("\nVictim Frequency Table (first 5 columns):")
    victim_freq = compute_frequency_table(df, 'victim')
    print(victim_freq.iloc[:, :5].to_string())
    
    print("\nShooter Proportion Table (first 5 columns):")
    shooter_freq = compute_frequency_table(df, 'shooter')
    shooter_prop = compute_proportion_table(shooter_freq)
    print(shooter_prop.iloc[:, :5].to_string())
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    print(f"""
Output files created:
  Tables:
    - {config.PROCESSED_DIR}/freq_table_victim.csv
    - {config.PROCESSED_DIR}/prop_table_victim.csv
    - {config.PROCESSED_DIR}/freq_table_shooter.csv
    - {config.PROCESSED_DIR}/prop_table_shooter.csv
  
  Figures:
    - {config.FIGURES_DIR}/heatmap_victim.png
    - {config.FIGURES_DIR}/heatmap_shooter.png
    - {config.FIGURES_DIR}/bar_top_clusters_victim.png
    - {config.FIGURES_DIR}/bar_top_clusters_shooter.png
""")


if __name__ == "__main__":
    main()
