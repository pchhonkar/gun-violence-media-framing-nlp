"""
Coreference resolution and context extraction module.

Processes articles to resolve coreferences and extract victim/shooter contexts.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
from tqdm import tqdm

from . import config
from . import utils_io


# =============================================================================
# KEYWORD CUES FOR CLASSIFICATION (as specified in assignment)
# =============================================================================

SHOOTER_CUES = [
    "suspect", "shooter", "gunman", "assailant", "attacker", "perpetrator",
    "opened fire", "fired", "arrested", "charged",
]

VICTIM_CUES = [
    "victim", "killed", "dead", "died", "injured", "wounded", "bystander",
    "student", "teacher", "hospitalized", "mourned",
]


def load_coref_model():
    """
    Load the coreference resolution model with spaCy.
    
    Returns:
        Tuple of (spaCy nlp model, fastcoref model)
    """
    print("Loading spaCy model...")
    # Use a smaller model for sentence segmentation if transformer not available
    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        print(f"Model {config.SPACY_MODEL} not found, trying en_core_web_sm...")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "No spaCy model found. Please run: python -m spacy download en_core_web_sm"
            )
    
    print("Loading fastcoref model...")
    try:
        from fastcoref import FCoref
        coref_model = FCoref(device="cpu")  # Use CPU for compatibility
    except ImportError:
        print("Warning: fastcoref not installed. Using fallback (no coref resolution).")
        coref_model = None
    except Exception as e:
        print(f"Warning: Failed to load fastcoref: {e}. Using fallback.")
        coref_model = None
    
    return nlp, coref_model


def resolve_coreferences_fastcoref(text: str, coref_model) -> str:
    """
    Resolve coreferences in text using fastcoref.
    
    Args:
        text: Input text
        coref_model: fastcoref FCoref model
        
    Returns:
        Text with pronouns replaced by their antecedents
    """
    if coref_model is None:
        return text
    
    try:
        # Get predictions
        preds = coref_model.predict(texts=[text])
        
        if preds and len(preds) > 0:
            # Get clusters for the first (only) text
            clusters = preds[0].get_clusters(as_strings=False)
            
            if not clusters:
                return text
            
            # Build replacement map: for each mention, find its antecedent
            # We'll replace pronouns with the first (usually most descriptive) mention
            replacements = []  # (start, end, replacement_text)
            
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                
                # Find the best antecedent (usually the first non-pronoun mention)
                text_spans = [(s, e, text[s:e]) for s, e in cluster]
                
                # Find best antecedent (longest non-pronoun, or first mention)
                antecedent = None
                for s, e, span_text in text_spans:
                    # Skip very short spans that are likely pronouns
                    if len(span_text) > 3 and span_text.lower() not in [
                        "he", "she", "him", "her", "his", "hers", "they", "them",
                        "their", "it", "its", "who", "whom", "which", "that"
                    ]:
                        antecedent = span_text
                        break
                
                if antecedent is None:
                    antecedent = text_spans[0][2]  # Fall back to first mention
                
                # Mark pronouns for replacement
                for s, e, span_text in text_spans:
                    if span_text.lower() in [
                        "he", "she", "him", "her", "his", "hers", "they", "them",
                        "their", "it", "its", "who", "whom"
                    ] and span_text != antecedent:
                        replacements.append((s, e, antecedent))
            
            # Sort replacements by position (reverse order to not mess up indices)
            replacements.sort(key=lambda x: x[0], reverse=True)
            
            # Apply replacements
            resolved_text = text
            for start, end, replacement in replacements:
                resolved_text = resolved_text[:start] + replacement + resolved_text[end:]
            
            return resolved_text
        
        return text
        
    except Exception as e:
        print(f"Warning: Coref resolution failed: {e}")
        return text


def resolve_coreferences(text: str, nlp, coref_model) -> Tuple[str, List[Dict]]:
    """
    Resolve coreferences in a text.
    
    Args:
        text: Input text
        nlp: spaCy model
        coref_model: Coreference model
        
    Returns:
        Tuple of (resolved text, list of coreference clusters)
    """
    resolved_text = resolve_coreferences_fastcoref(text, coref_model)
    
    # Note: We don't extract clusters separately here since fastcoref 
    # handles the full replacement
    clusters = []
    
    return resolved_text, clusters


def extract_sentences(text: str, nlp) -> List[str]:
    """
    Extract sentences from text using spaCy.
    
    Args:
        text: Input text
        nlp: spaCy model
        
    Returns:
        List of sentence strings
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def classify_sentence(sentence: str) -> Dict[str, bool]:
    """
    Classify a sentence as containing victim/shooter references.
    
    Args:
        sentence: Sentence text
        
    Returns:
        Dict with 'is_victim' and 'is_shooter' boolean flags
    """
    sentence_lower = sentence.lower()
    
    is_shooter = any(cue in sentence_lower for cue in SHOOTER_CUES)
    is_victim = any(cue in sentence_lower for cue in VICTIM_CUES)
    
    return {
        "is_victim": is_victim,
        "is_shooter": is_shooter,
    }


def process_single_article(
    article_id: str,
    outlet: str,
    raw_text: str,
    nlp,
    coref_model
) -> Dict:
    """
    Process a single article for coreference resolution and classification.
    
    Args:
        article_id: Unique article identifier
        outlet: News outlet name
        raw_text: Raw article text
        nlp: spaCy model
        coref_model: Coreference model
        
    Returns:
        Dictionary with processed results
    """
    # Resolve coreferences
    resolved_text, _ = resolve_coreferences(raw_text, nlp, coref_model)
    
    # Extract sentences from resolved text
    sentences = extract_sentences(resolved_text, nlp)
    
    # Classify each sentence
    victim_sentences = []
    shooter_sentences = []
    
    for sentence in sentences:
        classification = classify_sentence(sentence)
        
        if classification["is_victim"]:
            victim_sentences.append(sentence)
        if classification["is_shooter"]:
            shooter_sentences.append(sentence)
    
    return {
        "article_id": article_id,
        "outlet": outlet,
        "victim_sentences_resolved": victim_sentences,
        "shooter_sentences_resolved": shooter_sentences,
    }


def process_articles_for_coref(
    articles_df: pd.DataFrame,
    nlp=None,
    coref_model=None,
) -> List[Dict]:
    """
    Process all articles for coreference and extract entity contexts.
    
    Args:
        articles_df: DataFrame of articles with columns: article_id, outlet, raw_text
        nlp: spaCy model (loaded if not provided)
        coref_model: Coreference model (loaded if not provided)
        
    Returns:
        List of processed article dictionaries
    """
    # Load models if not provided
    if nlp is None or coref_model is None:
        nlp, coref_model = load_coref_model()
    
    results = []
    
    for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing articles"):
        result = process_single_article(
            article_id=row["article_id"],
            outlet=row["outlet"],
            raw_text=row["raw_text"],
            nlp=nlp,
            coref_model=coref_model,
        )
        results.append(result)
    
    return results


def save_jsonl(data: List[Dict], filepath: Path) -> None:
    """
    Save data to JSONL format (one JSON object per line).
    
    Args:
        data: List of dictionaries to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: Path) -> List[Dict]:
    """
    Load data from JSONL format.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of dictionaries
    """
    filepath = Path(filepath)
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data


def generate_task1_documentation() -> str:
    """
    Generate documentation for Task 1 (coreference resolution approach).
    <=300 words as specified.
    
    Returns:
        Documentation string in markdown format
    """
    doc = """# Task 1: Coreference Resolution and Context Extraction

## Approach

This module extracts victim and shooter sentences from news articles using a three-step pipeline.

### 1. Coreference Resolution
We use **fastcoref**, a transformer-based neural coreference model. It identifies mention clusters (text spans referring to the same entity) and replaces pronouns with their antecedents, producing "resolved" text.

### 2. Sentence Segmentation
We use **spaCy** to segment the resolved text into sentences, respecting grammatical boundaries.

### 3. Rule-Based Classification
Sentences are classified using keyword matching:

- **Shooter cues**: suspect, shooter, gunman, assailant, attacker, perpetrator, opened fire, fired, arrested, charged
- **Victim cues**: victim, killed, dead, died, injured, wounded, bystander, student, teacher, hospitalized, mourned

Sentences with both cue types appear in both output files.

## Challenges

1. **Coref Errors**: The model may incorrectly link mentions, especially with multiple entities (confusing shooter/victim pronouns).

2. **Unnamed Entities**: Early reports use "the suspect" without names, complicating resolution.

3. **Ambiguous Pronouns**: Sentences like "He shot him" require broader context.

4. **Cross-Sentence References**: Some references span sentences and may not fully resolve.

5. **Keyword Overlap**: Some sentences describe both victims and shooters simultaneously.

## Output
- `contexts_victims.jsonl`: Articles with victim-related sentences
- `contexts_shooters.jsonl`: Articles with shooter-related sentences

Each JSONL line contains: article_id, outlet, victim_sentences_resolved, shooter_sentences_resolved.
"""
    return doc


def main():
    """
    Main function to process articles for coreference resolution.
    Run with: python -m src.coref_contexts
    """
    print("=" * 60)
    print("Coreference Resolution and Context Extraction")
    print("=" * 60)
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    
    # Load articles
    articles_path = config.PROCESSED_DIR / "articles_master.csv"
    if not articles_path.exists():
        print(f"Error: Articles file not found at {articles_path}")
        print("Please run 'python -m src.load_articles' first.")
        return
    
    print(f"\nLoading articles from {articles_path}...")
    articles_df = utils_io.load_dataframe(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    # Load models
    print("\nLoading models...")
    nlp, coref_model = load_coref_model()
    
    # Process articles
    print("\nProcessing articles...")
    results = process_articles_for_coref(articles_df, nlp, coref_model)
    
    # Save JSONL files - each line has both victim and shooter sentences
    victim_path = config.PROCESSED_DIR / "contexts_victims.jsonl"
    shooter_path = config.PROCESSED_DIR / "contexts_shooters.jsonl"
    
    # Both files contain the same structure per assignment spec
    save_jsonl(results, victim_path)
    save_jsonl(results, shooter_path)
    
    print(f"\nSaved victim contexts to: {victim_path}")
    print(f"Saved shooter contexts to: {shooter_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    total_victim_sents = sum(len(r["victim_sentences_resolved"]) for r in results)
    total_shooter_sents = sum(len(r["shooter_sentences_resolved"]) for r in results)
    
    articles_with_victims = sum(1 for r in results if r["victim_sentences_resolved"])
    articles_with_shooters = sum(1 for r in results if r["shooter_sentences_resolved"])
    
    print(f"\nTotal victim sentences: {total_victim_sents}")
    print(f"Total shooter sentences: {total_shooter_sents}")
    print(f"Articles with victim sentences: {articles_with_victims}/{len(results)}")
    print(f"Articles with shooter sentences: {articles_with_shooters}/{len(results)}")
    
    # Per-outlet statistics
    print("\nPer-outlet breakdown:")
    outlet_stats = {}
    for result in results:
        outlet = result["outlet"]
        if outlet not in outlet_stats:
            outlet_stats[outlet] = {"victim": 0, "shooter": 0}
        outlet_stats[outlet]["victim"] += len(result["victim_sentences_resolved"])
        outlet_stats[outlet]["shooter"] += len(result["shooter_sentences_resolved"])
    
    for outlet, stats in sorted(outlet_stats.items()):
        print(f"  {outlet}: {stats['victim']} victim, {stats['shooter']} shooter sentences")
    
    # Generate and save documentation
    doc_path = config.REPORTS_DIR / "task1_doc.md"
    doc_content = generate_task1_documentation()
    utils_io.write_text_file(doc_path, doc_content)
    print(f"\nDocumentation saved to: {doc_path}")
    
    # Show sample output
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT")
    print("=" * 60)
    
    for result in results[:2]:
        print(f"\n{result['article_id']} ({result['outlet']}):")
        if result["victim_sentences_resolved"]:
            print(f"  Victim sentences ({len(result['victim_sentences_resolved'])}):")
            for sent in result["victim_sentences_resolved"][:2]:
                print(f"    - {sent[:100]}..." if len(sent) > 100 else f"    - {sent}")
        if result["shooter_sentences_resolved"]:
            print(f"  Shooter sentences ({len(result['shooter_sentences_resolved'])}):")
            for sent in result["shooter_sentences_resolved"][:2]:
                print(f"    - {sent[:100]}..." if len(sent) > 100 else f"    - {sent}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
