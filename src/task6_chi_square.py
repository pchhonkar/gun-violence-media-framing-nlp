"""
Task 6: Statistical Hypothesis Testing (Chi-Squared Test of Homogeneity)

This module tests whether the distribution of cluster usage differs significantly
across news outlets using Chi-Squared Test of Homogeneity.

Input:
    outputs/processed/frequency_table_victim.csv
    outputs/processed/frequency_table_shooter.csv
    (or outputs/processed/descriptions_with_clusters_refined.csv)

Outputs:
    - outputs/processed/task6_chi_square_results.csv
    - outputs/reports/task6_results.md
    - outputs/figures/task6_observed_vs_expected_*.png

Run with:
    python -m src.task6_chi_square
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple


# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

FREQ_VICTIM = PROCESSED_DIR / "frequency_table_victim.csv"
FREQ_SHOOTER = PROCESSED_DIR / "frequency_table_shooter.csv"
REFINED_DATA = PROCESSED_DIR / "descriptions_with_clusters_refined.csv"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
OUTLETS = ['CNN', 'Fox', 'NYT', 'WSJ']
ALPHA = 0.05
TOP_N = 3


# =============================================================================
# DATA LOADING
# =============================================================================
def load_frequency_table(entity_type: str) -> pd.DataFrame:
    """Load frequency table for an entity type."""
    path = FREQ_VICTIM if entity_type == 'victim' else FREQ_SHOOTER
    df = pd.read_csv(path, index_col=0)
    return df


def load_refined_data() -> pd.DataFrame:
    """Load the refined clustering data."""
    return pd.read_csv(REFINED_DATA)


# =============================================================================
# STEP 1: SELECT TOP CLUSTERS
# =============================================================================
def get_top_clusters(freq_table: pd.DataFrame, n: int = 3, 
                     exclude: List[str] = None) -> List[Tuple[str, int]]:
    """
    Get top N clusters by total count, excluding specified clusters.
    
    Args:
        freq_table: Frequency table with Total column
        n: Number of top clusters to return
        exclude: List of cluster names to exclude
    
    Returns:
        List of (cluster_name, total_count) tuples
    """
    if exclude is None:
        exclude = ['Noise/Unclustered']
    
    # Filter out excluded clusters
    filtered = freq_table[~freq_table.index.isin(exclude)]
    
    # Sort by Total and get top N
    if 'Total' in filtered.columns:
        top = filtered.nlargest(n, 'Total')
        return [(idx, row['Total']) for idx, row in top.iterrows()]
    else:
        # Calculate total from outlet columns
        filtered['_total'] = filtered[OUTLETS].sum(axis=1)
        top = filtered.nlargest(n, '_total')
        return [(idx, row['_total']) for idx, row in top.iterrows()]


# =============================================================================
# STEP 2: BUILD CONTINGENCY TABLES
# =============================================================================
def build_contingency_table(freq_table: pd.DataFrame, 
                           cluster_name: str) -> pd.DataFrame:
    """
    Build a 4x2 contingency table for a specific cluster.
    
    Rows: outlets (CNN, Fox, NYT, WSJ)
    Columns: [cluster_count, non_cluster_count]
    
    Args:
        freq_table: Full frequency table
        cluster_name: Name of the cluster to test
    
    Returns:
        DataFrame with contingency table
    """
    # Get cluster counts per outlet
    cluster_counts = freq_table.loc[cluster_name, OUTLETS]
    
    # Get total counts per outlet (sum of all clusters)
    outlet_totals = freq_table[OUTLETS].sum()
    
    # Calculate non-cluster counts
    non_cluster_counts = outlet_totals - cluster_counts
    
    # Build contingency table
    contingency = pd.DataFrame({
        'in_cluster': cluster_counts,
        'not_in_cluster': non_cluster_counts
    })
    
    return contingency


# =============================================================================
# STEP 3: CHI-SQUARE TEST
# =============================================================================
def run_chi_square_test(contingency: pd.DataFrame) -> Dict:
    """
    Run Chi-Square Test of Homogeneity.
    
    Args:
        contingency: 4x2 contingency table
    
    Returns:
        Dictionary with test results
    """
    # Run chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency.values)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected': expected,
        'observed': contingency.values,
        'reject_null': p_value < ALPHA
    }


# =============================================================================
# STEP 4: STANDARDIZED RESIDUALS
# =============================================================================
def compute_standardized_residuals(observed: np.ndarray, 
                                   expected: np.ndarray) -> np.ndarray:
    """
    Compute standardized residuals.
    
    residual = (observed - expected) / sqrt(expected)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals = (observed - expected) / np.sqrt(expected)
        residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
    return residuals


def identify_over_under_use(residuals: np.ndarray, 
                           outlets: List[str]) -> Tuple[str, str]:
    """
    Identify outlets with largest positive (overuse) and negative (underuse) residuals.
    
    Only looks at the 'in_cluster' column (first column).
    """
    # Residuals for 'in_cluster' column
    cluster_residuals = residuals[:, 0]
    
    overuse_idx = np.argmax(cluster_residuals)
    underuse_idx = np.argmin(cluster_residuals)
    
    return outlets[overuse_idx], outlets[underuse_idx]


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_observed_vs_expected(contingency: pd.DataFrame, 
                              expected: np.ndarray,
                              cluster_name: str,
                              entity_type: str,
                              output_path: Path) -> None:
    """Create bar plot of observed vs expected for each outlet."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(OUTLETS))
    width = 0.35
    
    observed = contingency['in_cluster'].values
    expected_vals = expected[:, 0]
    
    bars1 = ax.bar(x - width/2, observed, width, label='Observed', color='#2563eb')
    bars2 = ax.bar(x + width/2, expected_vals, width, label='Expected', color='#94a3b8')
    
    # Add value labels
    for bar, val in zip(bars1, observed):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, val in zip(bars2, expected_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Outlet', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{entity_type.capitalize()}: "{cluster_name[:40]}..." - Observed vs Expected',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(OUTLETS)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# DOCUMENTATION
# =============================================================================
def generate_results_markdown(all_results: List[Dict]) -> str:
    """Generate Task 6 results documentation."""
    
    doc = """# Task 6: Statistical Hypothesis Testing Results

## Overview

This analysis tests whether the distribution of cluster usage differs significantly 
across news outlets (CNN, Fox, NYT, WSJ) using the **Chi-Squared Test of Homogeneity**.

**Significance Level**: α = 0.05

## Method

For each of the top 3 clusters (by total count, excluding Noise/Unclustered):
1. Build a 4×2 contingency table (outlets × [in_cluster, not_in_cluster])
2. Run Chi-Square Test of Homogeneity
3. Compute standardized residuals to identify over/under-use by outlet

---

## Victim Cluster Tests

"""
    
    victim_results = [r for r in all_results if r['entity_type'] == 'victim']
    shooter_results = [r for r in all_results if r['entity_type'] == 'shooter']
    
    for i, r in enumerate(victim_results, 1):
        doc += f"""### Test {i}: {r['cluster_label']}

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "{r['cluster_label']}" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = {r['chi2']:.4f}
- df = {r['dof']}
- p-value = {r['p_value']:.6f}
- **Decision**: {"Reject H₀" if r['reject_null'] else "Fail to reject H₀"} (α = 0.05)

**Interpretation**: {r['interpretation']}

**Residual Analysis**:
- Overuse: **{r['overuse_outlet']}** (higher than expected)
- Underuse: **{r['underuse_outlet']}** (lower than expected)

---

"""
    
    doc += """## Shooter Cluster Tests

"""
    
    for i, r in enumerate(shooter_results, 1):
        doc += f"""### Test {i + len(victim_results)}: {r['cluster_label']}

**Null Hypothesis (H₀)**: The proportion of descriptions classified as "{r['cluster_label']}" 
is the same across all four outlets (CNN, Fox, NYT, WSJ).

**Alternative Hypothesis (H₁)**: At least one outlet has a different proportion.

**Results**:
- χ² = {r['chi2']:.4f}
- df = {r['dof']}
- p-value = {r['p_value']:.6f}
- **Decision**: {"Reject H₀" if r['reject_null'] else "Fail to reject H₀"} (α = 0.05)

**Interpretation**: {r['interpretation']}

**Residual Analysis**:
- Overuse: **{r['overuse_outlet']}** (higher than expected)
- Underuse: **{r['underuse_outlet']}** (lower than expected)

---

"""
    
    # Summary
    rejected_count = sum(1 for r in all_results if r['reject_null'])
    doc += f"""## Summary

| Entity Type | Cluster | χ² | p-value | Reject H₀? | Overuse | Underuse |
|-------------|---------|-----|---------|------------|---------|----------|
"""
    
    for r in all_results:
        doc += f"| {r['entity_type']} | {r['cluster_label'][:30]}... | {r['chi2']:.2f} | {r['p_value']:.4f} | {'Yes' if r['reject_null'] else 'No'} | {r['overuse_outlet']} | {r['underuse_outlet']} |\n"
    
    doc += f"""
**Overall**: {rejected_count} out of 6 tests showed statistically significant differences at α = 0.05.

## Conclusion

{"The Chi-Square tests reveal significant heterogeneity in how different outlets frame victims and shooters. " if rejected_count > 0 else "The tests did not reveal significant differences in framing across outlets. "}
The residual analysis helps identify which outlets drive these differences, showing patterns of over-use or under-use relative to expected frequencies under the null hypothesis.
"""
    
    return doc


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def run_analysis() -> List[Dict]:
    """Run the full Chi-Square analysis for both entity types."""
    
    all_results = []
    
    for entity_type in ['victim', 'shooter']:
        print(f"\n{'=' * 60}")
        print(f"ENTITY TYPE: {entity_type.upper()}")
        print('=' * 60)
        
        # Load frequency table
        freq_table = load_frequency_table(entity_type)
        
        # Step 1: Get top 3 clusters (excluding Noise/Unclustered)
        print("\n--- STEP 1: Top 3 Clusters (excluding Noise/Unclustered) ---")
        top_clusters = get_top_clusters(freq_table, n=TOP_N, exclude=['Noise/Unclustered'])
        
        for rank, (name, total) in enumerate(top_clusters, 1):
            print(f"  {rank}. {name} (Total: {total})")
        
        # Process each top cluster
        for rank, (cluster_name, cluster_total) in enumerate(top_clusters, 1):
            print(f"\n--- Testing Cluster: {cluster_name} ---")
            
            # Step 2: Build contingency table
            contingency = build_contingency_table(freq_table, cluster_name)
            print("\nContingency Table:")
            print(contingency.to_string())
            
            # Step 3: Run Chi-Square test
            test_result = run_chi_square_test(contingency)
            
            print(f"\nChi-Square Test Results:")
            print(f"  χ² = {test_result['chi2']:.4f}")
            print(f"  df = {test_result['dof']}")
            print(f"  p-value = {test_result['p_value']:.6f}")
            print(f"  Decision: {'REJECT H₀' if test_result['reject_null'] else 'FAIL TO REJECT H₀'} (α = {ALPHA})")
            
            # Step 4: Compute standardized residuals
            residuals = compute_standardized_residuals(
                test_result['observed'], 
                test_result['expected']
            )
            
            print("\nStandardized Residuals (in_cluster column):")
            residual_df = pd.DataFrame(
                {'Outlet': OUTLETS, 'Residual': residuals[:, 0]}
            )
            for _, row in residual_df.iterrows():
                symbol = "+" if row['Residual'] > 0 else ""
                print(f"  {row['Outlet']}: {symbol}{row['Residual']:.3f}")
            
            overuse, underuse = identify_over_under_use(residuals, OUTLETS)
            print(f"\n  Overuse outlet: {overuse}")
            print(f"  Underuse outlet: {underuse}")
            
            # Generate interpretation
            if test_result['reject_null']:
                interpretation = (
                    f"There is a statistically significant difference in how outlets use "
                    f"'{cluster_name}' framing. {overuse} uses this framing more than expected, "
                    f"while {underuse} uses it less than expected."
                )
            else:
                interpretation = (
                    f"No statistically significant difference in '{cluster_name}' usage across outlets. "
                    f"The observed variation could be due to chance."
                )
            
            # Store result
            result = {
                'entity_type': entity_type,
                'cluster_label': cluster_name,
                'chi2': test_result['chi2'],
                'dof': test_result['dof'],
                'p_value': test_result['p_value'],
                'reject_null': test_result['reject_null'],
                'overuse_outlet': overuse,
                'underuse_outlet': underuse,
                'interpretation': interpretation
            }
            all_results.append(result)
            
            # Create visualization
            fig_path = FIGURES_DIR / f"task6_observed_vs_expected_{entity_type}_{rank}.png"
            plot_observed_vs_expected(
                contingency, test_result['expected'],
                cluster_name, entity_type, fig_path
            )
            print(f"\nSaved figure: {fig_path}")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run Task 6: Chi-Square Hypothesis Testing."""
    print("=" * 60)
    print("Task 6: Statistical Hypothesis Testing")
    print("Chi-Squared Test of Homogeneity")
    print("=" * 60)
    
    # Check input files exist
    if not FREQ_VICTIM.exists() or not FREQ_SHOOTER.exists():
        print("\nERROR: Frequency tables not found. Run Task 5 first.")
        print(f"  Expected: {FREQ_VICTIM}")
        print(f"  Expected: {FREQ_SHOOTER}")
        return
    
    # Run analysis
    all_results = run_analysis()
    
    # Save results CSV
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    results_df = pd.DataFrame(all_results)
    results_path = PROCESSED_DIR / "task6_chi_square_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    
    # Generate and save documentation
    doc_content = generate_results_markdown(all_results)
    doc_path = REPORTS_DIR / "task6_results.md"
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    print(f"Saved: {doc_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TASK 6 COMPLETE")
    print("=" * 60)
    
    print("\nSUMMARY OF HYPOTHESIS TESTS:")
    print("-" * 60)
    print(f"{'Entity':<10} {'Cluster':<35} {'p-value':<12} {'Decision'}")
    print("-" * 60)
    
    for r in all_results:
        decision = "Reject H₀" if r['reject_null'] else "Fail to reject"
        cluster_short = r['cluster_label'][:32] + "..." if len(r['cluster_label']) > 32 else r['cluster_label']
        print(f"{r['entity_type']:<10} {cluster_short:<35} {r['p_value']:<12.6f} {decision}")
    
    rejected = sum(1 for r in all_results if r['reject_null'])
    print("-" * 60)
    print(f"\nTests with significant results (p < {ALPHA}): {rejected}/{len(all_results)}")
    
    print("\nFiles created:")
    print(f"  - {results_path}")
    print(f"  - {doc_path}")
    print(f"  - outputs/figures/task6_observed_vs_expected_*.png (6 files)")


if __name__ == "__main__":
    main()

