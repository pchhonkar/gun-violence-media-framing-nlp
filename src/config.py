"""
Configuration constants and parameters for the NLP pipeline.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data_100"
OUT_DIR = PROJECT_ROOT / "outputs"

# Output subdirectories
PROCESSED_DIR = OUT_DIR / "processed"
FIGURES_DIR = OUT_DIR / "figures"
REPORTS_DIR = OUT_DIR / "reports"

# News source directories
NEWS_SOURCES = {
    "cnn": DATA_DIR / "cnn_five_para",
    "fox": DATA_DIR / "FOX_five_para",
    "nyt": DATA_DIR / "NYT_five_para",
    "wsj": DATA_DIR / "WSJ_five_para",
}

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# MODEL NAMES
# =============================================================================
# spaCy model for NLP processing
SPACY_MODEL = "en_core_web_trf"  # transformer-based, most accurate
# Alternative: "en_core_web_lg" (faster, still good)

# Sentence embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # fast, good quality
# Alternative: "all-mpnet-base-v2" (slower, higher quality)

# Coreference model (for fastcoref)
COREF_MODEL = "biu-nlp/f-coref"

# =============================================================================
# CLUSTERING PARAMETERS
# =============================================================================
CLUSTERING_PARAMS = {
    # UMAP dimensionality reduction
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.1,
    "umap_metric": "cosine",
    
    # HDBSCAN / KMeans clustering
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples": 3,
    "kmeans_n_clusters": 10,  # placeholder, tune based on data
    
    # Agglomerative clustering
    "agg_n_clusters": None,  # set dynamically or use distance threshold
    "agg_distance_threshold": 0.5,
    "agg_linkage": "ward",
}

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================
# Context window for coreference (in sentences)
COREF_CONTEXT_WINDOW = 2

# Minimum description length (in tokens)
MIN_DESCRIPTION_LENGTH = 3

# Target entities for analysis (can be customized)
TARGET_ENTITIES = [
    "Trump",
    "Biden",
    # Add more as needed
]

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================
VIZ_PARAMS = {
    "figure_dpi": 150,
    "figure_format": "png",
    "colormap": "tab10",
    "font_size": 12,
}


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_source_label(source_key: str) -> str:
    """Get display label for a news source."""
    labels = {
        "cnn": "CNN",
        "fox": "Fox News",
        "nyt": "New York Times",
        "wsj": "Wall Street Journal",
    }
    return labels.get(source_key, source_key.upper())

