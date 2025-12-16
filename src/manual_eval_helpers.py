"""
Helpers for manual evaluation of clustering results.

Provides tools for reviewing clusters, generating evaluation templates,
and applying cluster refinements.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from . import config
from . import utils_io


def load_cluster_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load clustered descriptions and cluster summary.
    
    Returns:
        Tuple of (descriptions_df, summary_df)
    """
    desc_path = config.PROCESSED_DIR / "descriptions_with_clusters.csv"
    summary_path = config.PROCESSED_DIR / "cluster_summary.csv"
    
    if not desc_path.exists():
        raise FileNotFoundError(f"Descriptions file not found: {desc_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    descriptions_df = utils_io.load_dataframe(desc_path)
    summary_df = utils_io.load_dataframe(summary_path)
    
    return descriptions_df, summary_df


def print_cluster_overview(summary_df: pd.DataFrame, entity_type: str = None) -> None:
    """Print overview of all clusters."""
    if entity_type:
        summary_df = summary_df[summary_df['entity_type'] == entity_type]
    
    print("=" * 70)
    print(f"CLUSTER OVERVIEW{f' - {entity_type.upper()}' if entity_type else ''}")
    print("=" * 70)
    
    for _, row in summary_df.iterrows():
        print(f"\n[{row['entity_type'].upper()}] Cluster {row['cluster_id']}: {row['cluster_label']}")
        print(f"  Size: {row['size']} phrases")
        examples = row.get('examples', '')
        if examples:
            print(f"  Examples: {str(examples)[:200]}...")


def print_cluster_details(
    descriptions_df: pd.DataFrame,
    entity_type: str,
    cluster_id: int,
    n_examples: int = 20
) -> None:
    """Print detailed examples from a specific cluster."""
    filtered = descriptions_df[
        (descriptions_df['entity_type'] == entity_type) &
        (descriptions_df['cluster_id'] == cluster_id)
    ]
    
    if len(filtered) == 0:
        print(f"No data found for {entity_type} cluster {cluster_id}")
        return
    
    cluster_label = filtered['cluster_label'].iloc[0]
    
    print("=" * 70)
    print(f"CLUSTER DETAILS: {entity_type.upper()} - Cluster {cluster_id}")
    print(f"Label: {cluster_label}")
    print(f"Total phrases: {len(filtered)}")
    print("=" * 70)
    
    print(f"\nTop {min(n_examples, len(filtered))} examples:")
    print("-" * 50)
    
    for idx, (_, row) in enumerate(filtered.head(n_examples).iterrows()):
        print(f"{idx+1:3}. [{row['extraction_method']}] {row['description_phrase']}")
        print(f"     Source: {row['article_id']} | Outlet: {row['outlet']}")
    
    print("\nOutlet distribution:")
    outlet_counts = filtered['outlet'].value_counts()
    for outlet, count in outlet_counts.items():
        pct = count / len(filtered) * 100
        print(f"  {outlet}: {count} ({pct:.1f}%)")


def print_all_clusters_by_entity(
    descriptions_df: pd.DataFrame,
    entity_type: str,
    n_examples: int = 10
) -> None:
    """Print all clusters for an entity type with examples."""
    filtered = descriptions_df[descriptions_df['entity_type'] == entity_type]
    clusters = sorted(filtered['cluster_id'].unique())
    
    print("=" * 70)
    print(f"ALL CLUSTERS FOR {entity_type.upper()}")
    print(f"Total clusters: {len(clusters)}")
    print("=" * 70)
    
    for cluster_id in clusters:
        cluster_df = filtered[filtered['cluster_id'] == cluster_id]
        label = cluster_df['cluster_label'].iloc[0]
        
        print(f"\n{'='*50}")
        print(f"Cluster {cluster_id}: {label}")
        print(f"Size: {len(cluster_df)}")
        print("-" * 50)
        
        for idx, (_, row) in enumerate(cluster_df.head(n_examples).iterrows()):
            print(f"  {idx+1}. {row['description_phrase']}")


# =============================================================================
# REFINEMENT MAP APPLICATION
# =============================================================================

def load_refinement_map() -> Dict:
    """
    Load the cluster refinement mapping from JSON.
    
    Returns:
        Refinement mapping dictionary
    """
    map_path = config.PROCESSED_DIR / "cluster_refinement_map.json"
    
    if not map_path.exists():
        raise FileNotFoundError(f"Refinement map not found: {map_path}")
    
    return utils_io.load_json(map_path)


def match_pattern(phrase: str, pattern: str, pattern_type: str) -> bool:
    """
    Check if a phrase matches a pattern based on pattern type.
    
    Args:
        phrase: The phrase to check
        pattern: The pattern to match
        pattern_type: Type of matching (exact_lower, contains_lower, regex_lower)
        
    Returns:
        True if phrase matches pattern
    """
    phrase_lower = phrase.lower().strip()
    
    if pattern_type == "exact_lower":
        return phrase_lower == pattern.lower()
    elif pattern_type == "contains_lower":
        return pattern.lower() in phrase_lower
    elif pattern_type == "regex_lower":
        try:
            return bool(re.search(pattern, phrase_lower, re.IGNORECASE))
        except re.error:
            return False
    
    return False


def apply_refinement_map(
    df: pd.DataFrame,
    refinement_map: Dict = None
) -> pd.DataFrame:
    """
    Apply the refinement mapping to the descriptions DataFrame.
    
    Args:
        df: Original descriptions DataFrame with cluster assignments
        refinement_map: Refinement mapping (loaded if not provided)
        
    Returns:
        Refined DataFrame with updated cluster labels
    """
    if refinement_map is None:
        refinement_map = load_refinement_map()
    
    # Create a copy
    refined_df = df.copy()
    
    # Add refined_cluster_label column (initially copy of cluster_label)
    refined_df['refined_cluster_label'] = refined_df['cluster_label']
    
    # Track changes
    changes_made = 0
    
    # Apply shooter refinements
    shooter_refs = refinement_map.get('shooter_refinements', {})
    shooter_mask = refined_df['entity_type'] == 'shooter'
    
    for ref_name, ref_config in shooter_refs.items():
        new_label = ref_config.get('new_label')
        patterns = ref_config.get('patterns', [])
        pattern_type = ref_config.get('pattern_type', 'exact_lower')
        
        for idx in refined_df[shooter_mask].index:
            phrase = refined_df.loc[idx, 'description_phrase']
            
            for pattern in patterns:
                if match_pattern(phrase, pattern, pattern_type):
                    if refined_df.loc[idx, 'refined_cluster_label'] != new_label:
                        refined_df.loc[idx, 'refined_cluster_label'] = new_label
                        changes_made += 1
                    break
    
    # Apply victim refinements
    victim_refs = refinement_map.get('victim_refinements', {})
    victim_mask = refined_df['entity_type'] == 'victim'
    
    for ref_name, ref_config in victim_refs.items():
        new_label = ref_config.get('new_label')
        patterns = ref_config.get('patterns', [])
        pattern_type = ref_config.get('pattern_type', 'exact_lower')
        source_cluster_contains = ref_config.get('source_cluster_contains', '')
        
        for idx in refined_df[victim_mask].index:
            phrase = refined_df.loc[idx, 'description_phrase']
            current_label = refined_df.loc[idx, 'cluster_label']
            
            # Check if we should apply this refinement based on source cluster
            if source_cluster_contains and source_cluster_contains.lower() not in current_label.lower():
                continue
            
            for pattern in patterns:
                if match_pattern(phrase, pattern, pattern_type):
                    if refined_df.loc[idx, 'refined_cluster_label'] != new_label:
                        refined_df.loc[idx, 'refined_cluster_label'] = new_label
                        changes_made += 1
                    break
    
    print(f"Applied {changes_made} refinements")
    
    return refined_df


def create_refined_summary(refined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of the refined clusters.
    
    Args:
        refined_df: Refined DataFrame with refined_cluster_label
        
    Returns:
        Summary DataFrame
    """
    summaries = []
    
    for entity_type in ['victim', 'shooter']:
        entity_df = refined_df[refined_df['entity_type'] == entity_type]
        
        for label in entity_df['refined_cluster_label'].unique():
            cluster_df = entity_df[entity_df['refined_cluster_label'] == label]
            
            examples = cluster_df['description_phrase'].head(10).tolist()
            examples_str = "; ".join(examples)
            
            summaries.append({
                'entity_type': entity_type,
                'refined_cluster_label': label,
                'size': len(cluster_df),
                'examples': examples_str
            })
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(['entity_type', 'size'], ascending=[True, False])
    
    return summary_df


def generate_evaluation_template(
    descriptions_df: pd.DataFrame,
    summary_df: pd.DataFrame
) -> str:
    """Generate a markdown template for manual cluster evaluation."""
    template = """# Task 4: Manual Cluster Evaluation

## Instructions
For each cluster, evaluate:
1. **Lexical Coherence**: Do phrases share similar words/patterns? (Low/Medium/High)
2. **Semantic Coherence**: Do phrases have similar meaning/intent? (Low/Medium/High)
3. **Purity**: Are all phrases truly about the entity type? (Low/Medium/High)
4. **Misclassified Examples**: List any phrases that don't belong

---

"""
    
    for entity_type in ['victim', 'shooter']:
        entity_summary = summary_df[summary_df['entity_type'] == entity_type]
        entity_desc = descriptions_df[descriptions_df['entity_type'] == entity_type]
        
        template += f"# {entity_type.upper()} CLUSTERS\n\n"
        
        for _, row in entity_summary.iterrows():
            cluster_id = row.get('cluster_id', 'N/A')
            cluster_label = row.get('cluster_label', row.get('refined_cluster_label', 'Unknown'))
            size = row['size']
            
            label_col = 'cluster_label' if 'cluster_label' in entity_desc.columns else 'refined_cluster_label'
            cluster_phrases = entity_desc[
                entity_desc[label_col] == cluster_label
            ]['description_phrase'].head(15).tolist()
            
            template += f"## {cluster_label}\n\n"
            template += f"**Size:** {size} phrases\n\n"
            template += "**Sample Phrases:**\n"
            for i, phrase in enumerate(cluster_phrases, 1):
                template += f"{i}. {phrase}\n"
            template += "\n"
            
            template += "### Evaluation\n\n"
            template += "| Dimension | Rating | Notes |\n"
            template += "|-----------|--------|-------|\n"
            template += "| Lexical Coherence | ___ | |\n"
            template += "| Semantic Coherence | ___ | |\n"
            template += "| Purity | ___ | |\n"
            template += "\n---\n\n"
    
    return template


def compare_before_after(
    original_df: pd.DataFrame,
    refined_df: pd.DataFrame
) -> str:
    """
    Generate a comparison report between original and refined clusters.
    
    Args:
        original_df: Original DataFrame
        refined_df: Refined DataFrame
        
    Returns:
        Comparison report string
    """
    report = "# Before vs After Comparison\n\n"
    
    for entity_type in ['victim', 'shooter']:
        report += f"## {entity_type.upper()}\n\n"
        
        # Original counts
        orig_entity = original_df[original_df['entity_type'] == entity_type]
        orig_counts = orig_entity['cluster_label'].value_counts()
        
        # Refined counts
        ref_entity = refined_df[refined_df['entity_type'] == entity_type]
        ref_counts = ref_entity['refined_cluster_label'].value_counts()
        
        report += "### Original Clusters\n"
        for label, count in orig_counts.items():
            report += f"- {label}: {count}\n"
        
        report += "\n### Refined Clusters\n"
        for label, count in ref_counts.items():
            report += f"- {label}: {count}\n"
        
        report += "\n"
    
    return report


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function for manual evaluation and refinement.
    Run with: python -m src.manual_eval_helpers
    """
    print("=" * 60)
    print("Task 4: Manual Cluster Evaluation and Refinement")
    print("=" * 60)
    
    config.ensure_output_dirs()
    
    # Check input files
    desc_path = config.PROCESSED_DIR / "descriptions_with_clusters.csv"
    map_path = config.PROCESSED_DIR / "cluster_refinement_map.json"
    
    if not desc_path.exists():
        print(f"Error: Cluster file not found: {desc_path}")
        print("Please run 'python -m src.embed_cluster' first.")
        return
    
    # Load data
    print("\nLoading cluster data...")
    descriptions_df, summary_df = load_cluster_data()
    print(f"Loaded {len(descriptions_df)} descriptions")
    
    # Check if refinement map exists
    if not map_path.exists():
        print(f"\nNote: Refinement map not found at {map_path}")
        print("Creating default refinement map...")
        
        # Create default map
        default_map = {
            "metadata": {
                "description": "Manual cluster refinement mapping for Task 4",
                "notes": "Edit this file to define refinements"
            },
            "shooter_refinements": {},
            "victim_refinements": {},
            "preserved_clusters": []
        }
        utils_io.save_json(default_map, map_path)
        print(f"Created: {map_path}")
    
    # Load refinement map
    print("\nLoading refinement map...")
    refinement_map = load_refinement_map()
    
    # Apply refinements
    print("\nApplying refinements...")
    refined_df = apply_refinement_map(descriptions_df, refinement_map)
    
    # Save refined data
    refined_path = config.PROCESSED_DIR / "descriptions_with_clusters_refined.csv"
    utils_io.save_dataframe(refined_df, refined_path)
    print(f"\nSaved refined data to: {refined_path}")
    
    # Create refined summary
    refined_summary = create_refined_summary(refined_df)
    refined_summary_path = config.PROCESSED_DIR / "cluster_summary_refined.csv"
    utils_io.save_dataframe(refined_summary, refined_summary_path)
    print(f"Saved refined summary to: {refined_summary_path}")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("BEFORE VS AFTER COMPARISON")
    print("=" * 60)
    
    for entity_type in ['victim', 'shooter']:
        print(f"\n{entity_type.upper()}:")
        
        orig_entity = descriptions_df[descriptions_df['entity_type'] == entity_type]
        ref_entity = refined_df[refined_df['entity_type'] == entity_type]
        
        print("\n  Original clusters:")
        orig_counts = orig_entity['cluster_label'].value_counts()
        for label, count in orig_counts.head(10).items():
            print(f"    - {label}: {count}")
        
        print("\n  Refined clusters:")
        ref_counts = ref_entity['refined_cluster_label'].value_counts()
        for label, count in ref_counts.head(10).items():
            print(f"    - {label}: {count}")
    
    # Print refinement statistics
    print("\n" + "=" * 60)
    print("REFINEMENT STATISTICS")
    print("=" * 60)
    
    changed_mask = refined_df['cluster_label'] != refined_df['refined_cluster_label']
    n_changed = changed_mask.sum()
    n_total = len(refined_df)
    
    print(f"\nTotal descriptions: {n_total}")
    print(f"Descriptions refined: {n_changed} ({n_changed/n_total*100:.1f}%)")
    print(f"Descriptions unchanged: {n_total - n_changed}")
    
    # Unique labels before/after
    orig_labels = descriptions_df['cluster_label'].nunique()
    ref_labels = refined_df['refined_cluster_label'].nunique()
    
    print(f"\nUnique cluster labels (original): {orig_labels}")
    print(f"Unique cluster labels (refined): {ref_labels}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    print(f"""
Output files:
  - {refined_path}
  - {refined_summary_path}
  - {config.REPORTS_DIR / 'task4_manual_eval.md'}
""")


if __name__ == "__main__":
    main()
