"""
Article loading and preprocessing module.

Loads articles from data_100/ directory structure and creates a master DataFrame.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from . import config
from . import utils_io


def load_single_article(filepath: Path) -> Dict:
    """
    Load a single article from a text file.
    
    Args:
        filepath: Path to the article file
        
    Returns:
        Dictionary with article metadata and content
    """
    raw_text = utils_io.read_text_file(filepath)
    
    # Clean text: normalize whitespace but preserve paragraph structure
    cleaned_text = raw_text.strip()
    
    # Compute basic stats
    n_chars = len(cleaned_text)
    n_words = len(cleaned_text.split())
    
    return {
        "filename": filepath.name,
        "filepath": str(filepath),
        "raw_text": cleaned_text,
        "n_chars": n_chars,
        "n_words": n_words,
    }


def load_articles_from_source(source_key: str) -> List[Dict]:
    """
    Load all articles from a single news source.
    
    Args:
        source_key: Key identifying the news source (e.g., 'cnn', 'fox')
        
    Returns:
        List of article dictionaries
    """
    source_dir = config.NEWS_SOURCES.get(source_key)
    if source_dir is None:
        raise ValueError(f"Unknown source key: {source_key}")
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get all .txt files
    files = utils_io.list_files(source_dir, pattern="*.txt")
    
    articles = []
    for filepath in files:
        try:
            article = load_single_article(filepath)
            article["source_key"] = source_key
            articles.append(article)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return articles


def create_article_id(outlet: str, index: int) -> str:
    """
    Create a unique, stable article ID.
    
    Args:
        outlet: Outlet name (CNN, Fox, NYT, WSJ)
        index: Index number (1-based)
        
    Returns:
        Article ID string (e.g., "CNN_0001")
    """
    return f"{outlet}_{index:04d}"


def get_outlet_name(source_key: str) -> str:
    """
    Convert source key to standardized outlet name.
    
    Args:
        source_key: Source key (cnn, fox, nyt, wsj)
        
    Returns:
        Outlet name (CNN, Fox, NYT, WSJ)
    """
    mapping = {
        "cnn": "CNN",
        "fox": "Fox",
        "nyt": "NYT",
        "wsj": "WSJ",
    }
    return mapping.get(source_key, source_key.upper())


def load_all_articles() -> pd.DataFrame:
    """
    Load all articles from all news sources.
    
    Returns:
        DataFrame with columns: article_id, outlet, filename, filepath, raw_text, n_chars, n_words
    """
    all_articles = []
    
    for source_key in config.NEWS_SOURCES.keys():
        print(f"Loading articles from {source_key}...")
        articles = load_articles_from_source(source_key)
        
        outlet = get_outlet_name(source_key)
        
        # Assign article IDs - sort by filename for stability
        articles_sorted = sorted(articles, key=lambda x: x["filename"])
        
        for idx, article in enumerate(articles_sorted, start=1):
            article["article_id"] = create_article_id(outlet, idx)
            article["outlet"] = outlet
            all_articles.append(article)
    
    # Create DataFrame with specified column order
    df = pd.DataFrame(all_articles)
    
    # Reorder columns
    column_order = ["article_id", "outlet", "filename", "filepath", "raw_text", "n_chars", "n_words"]
    df = df[column_order]
    
    return df


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing (whitespace normalization, etc.).
    
    Args:
        text: Raw article text
        
    Returns:
        Cleaned text
    """
    # Normalize whitespace
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    
    # Remove empty lines but preserve paragraph breaks
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if line:
            cleaned_lines.append(line)
            prev_empty = False
        elif not prev_empty and cleaned_lines:
            cleaned_lines.append("")
            prev_empty = True
    
    return '\n'.join(cleaned_lines)


def get_article_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic statistics about the loaded articles.
    
    Args:
        df: DataFrame of articles
        
    Returns:
        DataFrame with statistics per source
    """
    stats = df.groupby("outlet").agg(
        n_articles=("article_id", "count"),
        avg_chars=("n_chars", "mean"),
        avg_words=("n_words", "mean"),
        total_words=("n_words", "sum"),
        min_words=("n_words", "min"),
        max_words=("n_words", "max"),
    ).round(1)
    
    return stats


def main():
    """
    Main function to load all articles and save to CSV.
    Run with: python -m src.load_articles
    """
    print("=" * 60)
    print("Loading articles from all sources...")
    print("=" * 60)
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    
    # Load all articles
    df = load_all_articles()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Counts per outlet
    print("\nArticles per outlet:")
    outlet_counts = df.groupby("outlet").size()
    for outlet, count in outlet_counts.items():
        print(f"  {outlet}: {count} articles")
    
    print(f"\nTotal articles: {len(df)}")
    
    # Statistics
    print("\nStatistics per outlet:")
    stats = get_article_stats(df)
    print(stats.to_string())
    
    # Save to CSV
    output_path = config.PROCESSED_DIR / "articles_master.csv"
    utils_io.save_dataframe(df, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Print sample
    print("\nSample articles:")
    print(df[["article_id", "outlet", "filename", "n_words"]].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
