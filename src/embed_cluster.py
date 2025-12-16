"""
Embedding and clustering pipeline module.

Embeds description phrases using SBERT and clusters with DBSCAN.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from . import utils_io


def load_embedding_model(model_name: str = None):
    """
    Load sentence embedding model.
    
    Args:
        model_name: Name of the model (default from config)
        
    Returns:
        Loaded SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer
    
    if model_name is None:
        model_name = config.EMBEDDING_MODEL
    
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    return model


def embed_descriptions(
    descriptions: List[str],
    model=None
) -> np.ndarray:
    """
    Generate embeddings for a list of descriptions.
    
    Args:
        descriptions: List of description strings
        model: Embedding model (loaded if not provided)
        
    Returns:
        Numpy array of embeddings (n_descriptions x embedding_dim)
    """
    if model is None:
        model = load_embedding_model()
    
    print(f"Embedding {len(descriptions)} descriptions...")
    embeddings = model.encode(
        descriptions,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def tune_dbscan(
    embeddings: np.ndarray,
    entity_type: str,
    eps_values: List[float] = None,
    min_samples_values: List[int] = None
) -> Tuple[float, int, str]:
    """
    Tune DBSCAN parameters to find good clustering.
    
    Args:
        embeddings: Embedding array
        entity_type: "victim" or "shooter"
        eps_values: List of eps values to try
        min_samples_values: List of min_samples values to try
        
    Returns:
        Tuple of (best_eps, best_min_samples, tuning_report)
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    
    if eps_values is None:
        eps_values = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.8]
    if min_samples_values is None:
        min_samples_values = [3, 5, 8]
    
    report_lines = [f"## DBSCAN Tuning Report - {entity_type.upper()}\n"]
    report_lines.append("| eps | min_samples | n_clusters | n_noise | noise_ratio | silhouette |")
    report_lines.append("|-----|-------------|------------|---------|-------------|------------|")
    
    best_score = -2
    best_params = (0.5, 5)
    best_n_clusters = 0
    
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_ratio = n_noise / len(labels)
            
            # Calculate silhouette score if we have valid clusters
            if n_clusters >= 2 and n_noise < len(labels) - 1:
                # Exclude noise points for silhouette
                mask = labels != -1
                if mask.sum() >= 2:
                    try:
                        sil_score = silhouette_score(
                            embeddings[mask], 
                            labels[mask], 
                            metric='cosine'
                        )
                    except:
                        sil_score = -1
                else:
                    sil_score = -1
            else:
                sil_score = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
                'silhouette': sil_score
            })
            
            report_lines.append(
                f"| {eps:.2f} | {min_samples} | {n_clusters} | {n_noise} | {noise_ratio:.2f} | {sil_score:.3f} |"
            )
            
            # Score: prefer more clusters, less noise, higher silhouette
            # Penalize all-noise or single-giant-cluster
            if n_clusters >= 2 and noise_ratio < 0.5:
                score = sil_score + 0.1 * min(n_clusters, 10) - 0.5 * noise_ratio
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
                    best_n_clusters = n_clusters
    
    # If no good params found, use defaults
    if best_score == -2:
        best_params = (0.5, 5)
        report_lines.append("\nNo optimal params found, using defaults (eps=0.5, min_samples=5)")
    else:
        report_lines.append(f"\n**Best params**: eps={best_params[0]}, min_samples={best_params[1]}")
        report_lines.append(f"**Clusters**: {best_n_clusters}, **Score**: {best_score:.3f}")
    
    return best_params[0], best_params[1], "\n".join(report_lines)


def cluster_embeddings(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> np.ndarray:
    """
    Cluster embeddings using DBSCAN.
    
    Args:
        embeddings: Embeddings to cluster
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        
    Returns:
        Cluster labels array
    """
    from sklearn.cluster import DBSCAN
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings)
    return labels


def assign_cluster_labels(
    df: pd.DataFrame,
    entity_type: str
) -> Dict[int, str]:
    """
    Assign human-readable labels to clusters based on heuristics.
    
    Args:
        df: DataFrame with cluster_id and description_phrase columns
        entity_type: "victim" or "shooter"
        
    Returns:
        Dictionary mapping cluster_id to label string
    """
    cluster_labels = {}
    
    # Define keyword patterns for labeling
    age_patterns = [
        r'\b\d+[\s-]?year[\s-]?old\b', r'\byoung\b', r'\bchild\b', r'\bchildren\b',
        r'\bkid\b', r'\bteen\b', r'\belderly\b', r'\bage\b', r'\baged\b'
    ]
    legal_patterns = [
        r'\barrested\b', r'\bcharged\b', r'\bcourt\b', r'\btrial\b', r'\bjudge\b',
        r'\blawyer\b', r'\battorney\b', r'\bprosecutor\b', r'\bdefendant\b',
        r'\bsentence\b', r'\bguilty\b', r'\binnocent\b', r'\bjail\b', r'\bprison\b'
    ]
    harm_patterns = [
        r'\bkilled\b', r'\bdied\b', r'\bdead\b', r'\binjured\b', r'\bwounded\b',
        r'\bshot\b', r'\bfatal\b', r'\bcritical\b', r'\bhospital\b', r'\btrauma\b'
    ]
    weapon_patterns = [
        r'\bgun\b', r'\brifle\b', r'\bweapon\b', r'\bfirearm\b', r'\bpistol\b',
        r'\bammunition\b', r'\bbullet\b'
    ]
    action_patterns = [
        r'\bopened fire\b', r'\bfired\b', r'\bshooting\b', r'\battack\b'
    ]
    
    unique_clusters = df['cluster_id'].unique()
    
    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster_id'] == cluster_id]
        phrases = cluster_df['description_phrase'].str.lower().tolist()
        combined_text = ' '.join(phrases)
        
        if cluster_id == -1:
            cluster_labels[cluster_id] = "Noise/Unclustered"
            continue
        
        # Count pattern matches
        age_count = sum(1 for p in phrases for pat in age_patterns if re.search(pat, p, re.I))
        legal_count = sum(1 for p in phrases for pat in legal_patterns if re.search(pat, p, re.I))
        harm_count = sum(1 for p in phrases for pat in harm_patterns if re.search(pat, p, re.I))
        weapon_count = sum(1 for p in phrases for pat in weapon_patterns if re.search(pat, p, re.I))
        action_count = sum(1 for p in phrases for pat in action_patterns if re.search(pat, p, re.I))
        
        total = len(phrases)
        
        # Assign label based on dominant pattern
        if age_count / total > 0.3:
            cluster_labels[cluster_id] = "Age references"
        elif legal_count / total > 0.3:
            cluster_labels[cluster_id] = "Legal/process framing"
        elif harm_count / total > 0.3:
            cluster_labels[cluster_id] = "Harm severity"
        elif weapon_count / total > 0.2:
            cluster_labels[cluster_id] = "Weapon references"
        elif action_count / total > 0.2:
            cluster_labels[cluster_id] = "Action descriptions"
        else:
            # Find most common meaningful word
            from collections import Counter
            words = []
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                        'for', 'of', 'with', 'by', 'was', 'were', 'is', 'are', 'been',
                        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                        'could', 'should', 'may', 'might', 'must', 'that', 'this', 'these',
                        'those', 'it', 'its', 'they', 'their', 'he', 'she', 'his', 'her'}
            for phrase in phrases:
                words.extend([w for w in phrase.split() if len(w) > 2 and w not in stopwords])
            
            if words:
                top_word = Counter(words).most_common(1)[0][0]
                cluster_labels[cluster_id] = f"Cluster {cluster_id}: {top_word}"
            else:
                cluster_labels[cluster_id] = f"Cluster {cluster_id}"
    
    return cluster_labels


def create_cluster_summary(
    df: pd.DataFrame,
    entity_type: str
) -> pd.DataFrame:
    """
    Create summary of clusters with examples.
    
    Args:
        df: DataFrame with cluster_id, cluster_label, description_phrase
        entity_type: "victim" or "shooter"
        
    Returns:
        Summary DataFrame
    """
    summaries = []
    
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        # Get top 10 example phrases
        examples = cluster_df['description_phrase'].head(10).tolist()
        examples_str = "; ".join(examples)
        
        summaries.append({
            'entity_type': entity_type,
            'cluster_id': cluster_id,
            'cluster_label': cluster_df['cluster_label'].iloc[0],
            'size': len(cluster_df),
            'examples': examples_str
        })
    
    return pd.DataFrame(summaries)


def reduce_to_2d(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.
    
    Args:
        embeddings: High-dimensional embeddings
        method: "umap" or "tsne"
        
    Returns:
        2D coordinates array
    """
    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=config.RANDOM_SEED
        )
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            random_state=config.RANDOM_SEED
        )
    
    coords_2d = reducer.fit_transform(embeddings)
    return coords_2d


def plot_clusters(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    cluster_labels_map: Dict[int, str],
    entity_type: str,
    output_path: Path
) -> None:
    """
    Plot UMAP/t-SNE clusters.
    
    Args:
        coords_2d: 2D coordinates
        labels: Cluster labels
        cluster_labels_map: Mapping from cluster_id to label string
        entity_type: "victim" or "shooter"
        output_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = sorted(set(labels))
    
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        
        if cluster_id == -1:
            # Noise points - gray, smaller
            plt.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c='lightgray',
                s=20,
                alpha=0.5,
                label=f"Noise (n={mask.sum()})"
            )
        else:
            label_str = cluster_labels_map.get(cluster_id, f"Cluster {cluster_id}")
            plt.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[colors[idx]],
                s=40,
                alpha=0.7,
                label=f"{label_str} (n={mask.sum()})"
            )
    
    plt.title(f"UMAP Clusters - {entity_type.capitalize()} Descriptions", fontsize=14)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")


def generate_task3_documentation() -> str:
    """
    Generate documentation for Task 3 (embedding & clustering).
    <=300 words as specified.
    
    Returns:
        Documentation string in markdown format
    """
    doc = """# Task 3: Embedding and Clustering Documentation

## Approach

### Embedding
We use **Sentence-BERT** (all-MiniLM-L6-v2) to embed description phrases into dense vectors. This model captures semantic similarity, grouping phrases with similar meanings regardless of exact wording.

### Clustering
We apply **DBSCAN** (Density-Based Spatial Clustering) separately for victim and shooter descriptions.

**Why DBSCAN:**
- No need to pre-specify number of clusters
- Identifies noise/outliers naturally
- Handles varying cluster densities

**Parameter Tuning:**
- Grid search over eps (0.25-0.8) and min_samples (3-8)
- Optimize for: multiple clusters, low noise ratio, high silhouette score
- Avoid: all-noise or single-giant-cluster outcomes

### Cluster Labeling
Heuristic-based labeling using keyword patterns:
- **Age references**: age, year-old, child, young, elderly
- **Legal/process framing**: arrested, charged, court, trial
- **Harm severity**: killed, died, injured, wounded, hospital
- **Weapon references**: gun, rifle, weapon, firearm
- **Action descriptions**: opened fire, fired, shooting

Default: most frequent non-stopword in cluster.

### Visualization
**UMAP** projects embeddings to 2D, preserving local and global structure. Points colored by cluster, noise shown in gray.

## Strengths
1. Semantic embeddings capture meaning beyond keywords
2. DBSCAN naturally separates noise from meaningful clusters
3. Automated labeling provides interpretable cluster names

## Limitations
1. DBSCAN sensitive to eps parameter
2. Short phrases may have noisy embeddings
3. Heuristic labels may miss domain-specific patterns
4. High-dimensional embeddings may lose nuance in 2D projection

## Output
- `descriptions_with_clusters.csv`: Original data + cluster assignments
- `cluster_summary.csv`: Cluster statistics and examples
- UMAP visualizations for each entity type
"""
    return doc


def process_entity_type(
    df: pd.DataFrame,
    entity_type: str,
    embeddings: np.ndarray,
    tuning_reports: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[int, str]]:
    """
    Process clustering for one entity type.
    
    Args:
        df: DataFrame filtered to entity_type
        entity_type: "victim" or "shooter"
        embeddings: Embeddings for this entity type
        tuning_reports: List to append tuning report to
        
    Returns:
        Tuple of (df with clusters, summary df, labels, cluster_labels_map)
    """
    print(f"\n{'='*40}")
    print(f"Processing {entity_type} descriptions")
    print(f"{'='*40}")
    
    # Tune DBSCAN parameters
    print("Tuning DBSCAN parameters...")
    best_eps, best_min_samples, tuning_report = tune_dbscan(
        embeddings, entity_type
    )
    tuning_reports.append(tuning_report)
    
    print(f"Best params: eps={best_eps}, min_samples={best_min_samples}")
    
    # Cluster with best params
    labels = cluster_embeddings(embeddings, eps=best_eps, min_samples=best_min_samples)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
    
    # Add cluster_id to dataframe
    df = df.copy()
    df['cluster_id'] = labels
    
    # Assign cluster labels
    cluster_labels_map = assign_cluster_labels(df, entity_type)
    df['cluster_label'] = df['cluster_id'].map(cluster_labels_map)
    
    # Create summary
    summary_df = create_cluster_summary(df, entity_type)
    
    return df, summary_df, labels, cluster_labels_map


def main():
    """
    Main function to embed and cluster descriptions.
    Run with: python -m src.embed_cluster
    """
    print("=" * 60)
    print("Embedding and Clustering Pipeline")
    print("=" * 60)
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    
    # Check input file exists
    input_path = config.PROCESSED_DIR / "descriptions.csv"
    if not input_path.exists():
        print(f"Error: Descriptions file not found at {input_path}")
        print("Please run 'python -m src.extract_descriptions' first.")
        return
    
    # Load descriptions
    print(f"\nLoading descriptions from {input_path}...")
    df = utils_io.load_dataframe(input_path)
    print(f"Loaded {len(df)} descriptions")
    
    # Load embedding model
    model = load_embedding_model()
    
    # Split by entity type
    victim_df = df[df['entity_type'] == 'victim'].reset_index(drop=True)
    shooter_df = df[df['entity_type'] == 'shooter'].reset_index(drop=True)
    
    print(f"\nVictim descriptions: {len(victim_df)}")
    print(f"Shooter descriptions: {len(shooter_df)}")
    
    # Embed all descriptions
    print("\nEmbedding victim descriptions...")
    victim_embeddings = embed_descriptions(
        victim_df['description_phrase'].tolist(),
        model
    )
    
    print("\nEmbedding shooter descriptions...")
    shooter_embeddings = embed_descriptions(
        shooter_df['description_phrase'].tolist(),
        model
    )
    
    # Process each entity type
    tuning_reports = []
    
    # Process victims
    victim_df_clustered, victim_summary, victim_labels, victim_labels_map = process_entity_type(
        victim_df, "victim", victim_embeddings, tuning_reports
    )
    
    # Process shooters
    shooter_df_clustered, shooter_summary, shooter_labels, shooter_labels_map = process_entity_type(
        shooter_df, "shooter", shooter_embeddings, tuning_reports
    )
    
    # Combine results
    combined_df = pd.concat([victim_df_clustered, shooter_df_clustered], ignore_index=True)
    combined_summary = pd.concat([victim_summary, shooter_summary], ignore_index=True)
    
    # Save descriptions with clusters
    output_path = config.PROCESSED_DIR / "descriptions_with_clusters.csv"
    utils_io.save_dataframe(combined_df, output_path)
    print(f"\nSaved clustered descriptions to: {output_path}")
    
    # Save cluster summary
    summary_path = config.PROCESSED_DIR / "cluster_summary.csv"
    utils_io.save_dataframe(combined_summary, summary_path)
    print(f"Saved cluster summary to: {summary_path}")
    
    # Create UMAP visualizations
    print("\nGenerating UMAP visualizations...")
    
    # Victim UMAP
    print("Reducing victim embeddings to 2D...")
    victim_2d = reduce_to_2d(victim_embeddings, method="umap")
    plot_clusters(
        victim_2d,
        victim_labels,
        victim_labels_map,
        "victim",
        config.FIGURES_DIR / "umap_clusters_victim.png"
    )
    
    # Shooter UMAP
    print("Reducing shooter embeddings to 2D...")
    shooter_2d = reduce_to_2d(shooter_embeddings, method="umap")
    plot_clusters(
        shooter_2d,
        shooter_labels,
        shooter_labels_map,
        "shooter",
        config.FIGURES_DIR / "umap_clusters_shooter.png"
    )
    
    # Save tuning report
    tuning_path = config.REPORTS_DIR / "dbscan_tuning.md"
    tuning_content = "# DBSCAN Parameter Tuning Report\n\n" + "\n\n".join(tuning_reports)
    utils_io.write_text_file(tuning_path, tuning_content)
    print(f"\nSaved tuning report to: {tuning_path}")
    
    # Save task documentation
    doc_path = config.REPORTS_DIR / "task3_doc.md"
    doc_content = generate_task3_documentation()
    utils_io.write_text_file(doc_path, doc_content)
    print(f"Saved documentation to: {doc_path}")
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    print("\nCluster Summary:")
    print(combined_summary[['entity_type', 'cluster_id', 'cluster_label', 'size']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
