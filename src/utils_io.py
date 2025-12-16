"""
File I/O utilities for the NLP pipeline.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


def read_text_file(
    filepath: Union[str, Path], 
    encoding: str = "utf-8",
    fallback_encoding: str = "latin-1"
) -> str:
    """
    Read a text file and return its contents.
    Tries primary encoding first, falls back to secondary if that fails.
    
    Args:
        filepath: Path to the text file
        encoding: Primary file encoding (default: utf-8)
        fallback_encoding: Fallback encoding if primary fails
        
    Returns:
        File contents as a string
    """
    filepath = Path(filepath)
    
    # Try primary encoding first
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    
    # Fallback to secondary encoding
    try:
        with open(filepath, 'r', encoding=fallback_encoding) as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"Failed to read {filepath} with both {encoding} and {fallback_encoding}"
        )


def write_text_file(
    filepath: Union[str, Path], 
    content: str, 
    encoding: str = "utf-8"
) -> None:
    """
    Write content to a text file.
    
    Args:
        filepath: Path to the output file
        content: Content to write
        encoding: File encoding (default: utf-8)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(content)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable)
        filepath: Path to the output file
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to pickle
        filepath: Path to the output file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Unpickled object
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_dataframe(
    df: pd.DataFrame, 
    filepath: Union[str, Path], 
    index: bool = False
) -> None:
    """
    Save a DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        filepath: Path to the output file
        index: Whether to include the index
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=index, encoding='utf-8')


def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a DataFrame from CSV.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    return pd.read_csv(filepath, encoding='utf-8')


def list_files(
    directory: Union[str, Path], 
    pattern: str = "*.txt"
) -> List[Path]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for matching files
        
    Returns:
        List of matching file paths, sorted alphabetically
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = sorted(directory.glob(pattern))
    return files
