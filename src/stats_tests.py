"""
Statistical hypothesis testing module.

Performs Chi-squared tests of homogeneity to compare cluster distributions
across news outlets.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from . import utils_io


def get_top_clusters(
    df: pd.DataFrame,
    entity_type: str,
    n: int = 3,
    exclude_noise: bool = True
) -> List[Tuple[int, str, int]]:
    """
    Get top N most frequent clusters for an entity type.
    
    Args:
        df: Descriptions DataFrame with clusters
        entity_type: "victim" or "shooter"
        n: Number of top clusters
        exclude_noise: Whether to exclude noise cluster (id=-1)
        
    Returns:
        List of tuples (cluster_id, cluster_label, count)
    """
    filtered = df[df['entity_type'] == entity_type]
    
    if exclude_noise:
        filtered = filtered[filtered['cluster_id'] != -1]
    
    # Count by cluster
    cluster_counts = filtered.groupby(['cluster_id', 'cluster_label']).size()
    cluster_counts = cluster_counts.reset_index(name='count')
    cluster_counts = cluster_counts.sort_values('count', ascending=False)
    
    top_clusters = []
    for _, row in cluster_counts.head(n).iterrows():
        top_clusters.append((
            int(row['cluster_id']),
            row['cluster_label'],
            int(row['count'])
        ))
    
    return top_clusters


def build_contingency_table(
    df: pd.DataFrame,
    entity_type: str,
    cluster_id: int
) -> pd.DataFrame:
    """
    Build a 4x2 contingency table for a specific cluster.
    
    Rows: outlets (CNN, Fox, NYT, WSJ)
    Cols: [count_in_cluster, count_not_in_cluster]
    
    Args:
        df: Descriptions DataFrame
        entity_type: "victim" or "shooter"
        cluster_id: Cluster ID to test
        
    Returns:
        Contingency table DataFrame
    """
    # Filter to entity type
    filtered = df[df['entity_type'] == entity_type]
    
    # Define outlets
    outlets = ['CNN', 'Fox', 'NYT', 'WSJ']
    
    # Build contingency table
    data = []
    for outlet in outlets:
        outlet_data = filtered[filtered['outlet'] == outlet]
        in_cluster = (outlet_data['cluster_id'] == cluster_id).sum()
        not_in_cluster = (outlet_data['cluster_id'] != cluster_id).sum()
        data.append({
            'outlet': outlet,
            'in_cluster': in_cluster,
            'not_in_cluster': not_in_cluster
        })
    
    contingency_df = pd.DataFrame(data)
    contingency_df = contingency_df.set_index('outlet')
    
    return contingency_df


def run_chi_square_test(contingency_table: pd.DataFrame) -> Dict:
    """
    Run Chi-squared test of homogeneity.
    
    Args:
        contingency_table: 4x2 contingency table
        
    Returns:
        Dictionary with chi2, p_value, dof, expected frequencies
    """
    # Convert to numpy array for scipy
    observed = contingency_table.values
    
    # Run chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected': expected
    }


def generate_interpretation(
    cluster_label: str,
    chi2: float,
    p_value: float,
    reject: bool,
    contingency_table: pd.DataFrame
) -> str:
    """
    Generate a brief interpretation of the test result.
    
    Args:
        cluster_label: Label of the cluster tested
        chi2: Chi-squared statistic
        p_value: P-value
        reject: Whether to reject null hypothesis
        contingency_table: The contingency table
        
    Returns:
        1-2 sentence interpretation
    """
    if reject:
        # Find which outlet has highest/lowest proportion
        total_per_outlet = contingency_table.sum(axis=1)
        prop_in_cluster = contingency_table['in_cluster'] / total_per_outlet * 100
        
        highest_outlet = prop_in_cluster.idxmax()
        lowest_outlet = prop_in_cluster.idxmin()
        highest_pct = prop_in_cluster[highest_outlet]
        lowest_pct = prop_in_cluster[lowest_outlet]
        
        interpretation = (
            f"Significant difference in '{cluster_label}' usage across outlets (p={p_value:.4f}). "
            f"{highest_outlet} uses this cluster most ({highest_pct:.1f}%), "
            f"while {lowest_outlet} uses it least ({lowest_pct:.1f}%)."
        )
    else:
        interpretation = (
            f"No significant difference in '{cluster_label}' usage across outlets (p={p_value:.4f}). "
            f"All outlets use this cluster at similar rates."
        )
    
    return interpretation


def run_tests_for_entity_type(
    df: pd.DataFrame,
    entity_type: str,
    n_clusters: int = 3,
    alpha: float = 0.05
) -> List[Dict]:
    """
    Run chi-squared tests for top clusters of an entity type.
    
    Args:
        df: Descriptions DataFrame
        entity_type: "victim" or "shooter"
        n_clusters: Number of top clusters to test
        alpha: Significance level
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Get top clusters
    top_clusters = get_top_clusters(df, entity_type, n=n_clusters, exclude_noise=True)
    
    print(f"\n{entity_type.upper()} - Testing top {len(top_clusters)} clusters:")
    
    for cluster_id, cluster_label, cluster_count in top_clusters:
        print(f"\n  Cluster {cluster_id}: {cluster_label} (n={cluster_count})")
        
        # Build contingency table
        contingency = build_contingency_table(df, entity_type, cluster_id)
        
        print(f"    Contingency table:")
        print(f"    {contingency.to_string().replace(chr(10), chr(10) + '    ')}")
        
        # Run chi-squared test
        test_result = run_chi_square_test(contingency)
        
        chi2 = test_result['chi2']
        p_value = test_result['p_value']
        dof = test_result['dof']
        reject = p_value < alpha
        
        print(f"    Chi2={chi2:.4f}, p={p_value:.4f}, dof={dof}")
        print(f"    Reject H0 at alpha={alpha}: {reject}")
        
        # Generate interpretation
        interpretation = generate_interpretation(
            cluster_label, chi2, p_value, reject, contingency
        )
        
        results.append({
            'entity_type': entity_type,
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_size': cluster_count,
            'chi2': round(chi2, 4),
            'p_value': round(p_value, 6),
            'dof': dof,
            'reject_0_05': reject,
            'interpretation': interpretation
        })
    
    return results


def generate_hypotheses_report(results: List[Dict], alpha: float = 0.05) -> str:
    """
    Generate the task6_hypotheses.md report.
    
    Args:
        results: List of test result dictionaries
        alpha: Significance level
        
    Returns:
        Markdown report string
    """
    report = """# Task 6: Chi-Squared Test Hypotheses and Results

## Overview
We test whether the distribution of each top cluster differs significantly across news outlets (CNN, Fox, NYT, WSJ) using the Chi-squared test of homogeneity.

## Statistical Framework

**Test**: Chi-squared Test of Homogeneity  
**Significance Level**: α = 0.05  
**Contingency Table Structure**: 4 rows (outlets) × 2 columns (in_cluster, not_in_cluster)

---

"""
    
    # Group by entity type
    for entity_type in ['victim', 'shooter']:
        entity_results = [r for r in results if r['entity_type'] == entity_type]
        
        if not entity_results:
            continue
        
        report += f"# {entity_type.upper()} CLUSTERS\n\n"
        
        for result in entity_results:
            cluster_label = result['cluster_label']
            chi2 = result['chi2']
            p_value = result['p_value']
            dof = result['dof']
            reject = result['reject_0_05']
            interpretation = result['interpretation']
            
            report += f"## Cluster: {cluster_label}\n\n"
            
            # Null hypothesis
            report += "### Null Hypothesis (H₀)\n\n"
            report += f"The proportion of {entity_type} descriptions in the \"{cluster_label}\" cluster "
            report += "is the same across all four news outlets (CNN, Fox News, NYT, WSJ).\n\n"
            
            # Alternative hypothesis
            report += "### Alternative Hypothesis (H₁)\n\n"
            report += f"The proportion of {entity_type} descriptions in the \"{cluster_label}\" cluster "
            report += "differs across at least two news outlets.\n\n"
            
            # Results
            report += "### Test Results\n\n"
            report += f"| Statistic | Value |\n"
            report += f"|-----------|-------|\n"
            report += f"| Chi-squared (χ²) | {chi2:.4f} |\n"
            report += f"| Degrees of Freedom | {dof} |\n"
            report += f"| P-value | {p_value:.6f} |\n"
            report += f"| Reject H₀ at α=0.05 | {'Yes' if reject else 'No'} |\n\n"
            
            # Conclusion
            report += "### Conclusion\n\n"
            if reject:
                report += f"**Reject the null hypothesis** (p = {p_value:.4f} < 0.05).\n\n"
                report += f"There is statistically significant evidence that the usage of the \"{cluster_label}\" "
                report += f"cluster differs across news outlets for {entity_type} descriptions.\n\n"
            else:
                report += f"**Fail to reject the null hypothesis** (p = {p_value:.4f} ≥ 0.05).\n\n"
                report += f"There is insufficient evidence to conclude that the usage of the \"{cluster_label}\" "
                report += f"cluster differs across news outlets for {entity_type} descriptions.\n\n"
            
            report += f"**Interpretation**: {interpretation}\n\n"
            report += "---\n\n"
    
    # Summary table
    report += "# SUMMARY TABLE\n\n"
    report += "| Entity Type | Cluster | χ² | p-value | Reject H₀? |\n"
    report += "|-------------|---------|----|---------|-----------|\n"
    
    for result in results:
        reject_str = "✓" if result['reject_0_05'] else "✗"
        report += f"| {result['entity_type']} | {result['cluster_label'][:30]} | {result['chi2']:.2f} | {result['p_value']:.4f} | {reject_str} |\n"
    
    report += "\n"
    
    # Overall conclusions
    n_rejected = sum(1 for r in results if r['reject_0_05'])
    n_total = len(results)
    
    report += "# OVERALL CONCLUSIONS\n\n"
    report += f"Out of {n_total} tests conducted:\n"
    report += f"- **{n_rejected}** showed significant differences (p < 0.05)\n"
    report += f"- **{n_total - n_rejected}** showed no significant differences\n\n"
    
    if n_rejected > 0:
        report += "The significant results suggest that news outlets differ in how they frame "
        report += "certain aspects of victims and/or shooters in mass shooting coverage.\n"
    else:
        report += "The lack of significant results suggests that news outlets use similar "
        report += "language patterns when describing victims and shooters.\n"
    
    return report


def main():
    """
    Main function to run statistical tests.
    Run with: python -m src.stats_tests
    """
    print("=" * 60)
    print("Statistical Hypothesis Testing")
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
    
    # Run tests for each entity type
    all_results = []
    
    for entity_type in ['victim', 'shooter']:
        results = run_tests_for_entity_type(df, entity_type, n_clusters=3)
        all_results.extend(results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results CSV
    results_path = config.PROCESSED_DIR / "chi_square_results.csv"
    utils_io.save_dataframe(results_df, results_path)
    print(f"\nSaved results to: {results_path}")
    
    # Generate and save hypotheses report
    report = generate_hypotheses_report(all_results)
    report_path = config.REPORTS_DIR / "task6_hypotheses.md"
    utils_io.write_text_file(report_path, report)
    print(f"Saved hypotheses report to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nResults by entity type:")
    print(results_df[['entity_type', 'cluster_label', 'chi2', 'p_value', 'reject_0_05']].to_string(index=False))
    
    n_rejected = results_df['reject_0_05'].sum()
    print(f"\nSignificant results (p < 0.05): {n_rejected} / {len(results_df)}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
