# Task 3: Embedding and Clustering Documentation

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
