"""
Entity description extraction module.

Extracts descriptive phrases from victim/shooter sentences using
entity-linked dependency parsing and POS tagging, with quality filtering.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import spacy
from tqdm import tqdm

from . import config
from . import utils_io


# =============================================================================
# CUE WORDS AND KEYWORDS
# =============================================================================

# Cue words for anchor detection
SHOOTER_CUE_WORDS = {
    "suspect", "shooter", "gunman", "assailant", "attacker", "perpetrator",
    "accused", "defendant", "killer", "murderer",
}

VICTIM_CUE_WORDS = {
    "victim", "victims", "bystander", "student", "students", "teacher", "teachers",
    "child", "children", "people", "person", "kid", "kids",
}

# Lemmas for anchor detection
SHOOTER_ANCHOR_LEMMAS = {"suspect", "shooter", "gunman", "assailant", "attacker", "perpetrator", "killer"}
VICTIM_ANCHOR_LEMMAS = {"victim", "bystander", "student", "teacher", "child", "kid", "person", "people"}

# Harm terms for victim anchor detection
HARM_TERMS = {"killed", "injured", "wounded", "dead", "died", "shot", "hospitalized", "critical", "fatal"}

# Location-only terms to filter
LOCATION_ONLY_TERMS = {"school", "elementary", "campus", "building", "street", "town", "city", "county", "state", "community", "neighborhood", "district"}

# =============================================================================
# HIGH-VALUE KEYWORDS (updated - removed school/elementary from victim)
# =============================================================================

SHOOTER_KEYWORDS = {
    "shooter", "suspect", "gunman", "assailant", "attacker", "perpetrator",
    "arrested", "charged", "opened fire", "fired", "rifle", "handgun", "weapon",
    "killed", "shot", "dead", "wounded", "murder", "shooting", "attack",
    "custody", "detained", "accused", "indicted", "convicted", "sentence",
    "armed", "gun", "firearm", "ammunition", "bullet", "trigger",
    "school shooting", "mass shooting", "deadly shooting", "shooting spree",
}

VICTIM_KEYWORDS = {
    "victim", "killed", "dead", "died", "injured", "wounded", "bystander",
    "student", "teacher", "child", "children", "hospital", "critical", "survivor",
    "mourning", "family", "hospitalized", "fatal", "deceased", "slain",
    "shooting victim", "casualty", "casualties", "trauma", "grieving",
    "young", "innocent", "loss", "tragedy", "tragic",
    "school shooting", "mass shooting", "deadly shooting",
}

# =============================================================================
# REPORTING VERBS TO FILTER
# =============================================================================

REPORTING_VERBS = {
    "say", "tell", "report", "ask", "add", "note", "claim", "according",
    "describe", "explain", "recall", "write", "tweet", "post", "share", "call",
    "state", "announce", "confirm", "reveal", "suggest", "indicate", "mention",
    "comment", "respond", "reply", "answer", "speak", "talk", "discuss",
    "believe", "think", "know", "feel", "want", "need", "seem", "appear",
}

# =============================================================================
# FRAME HINT KEYWORDS
# =============================================================================

FRAME_KEYWORDS = {
    "harm": {"killed", "dead", "died", "injured", "wounded", "shot", "hospitalized", 
             "critical", "fatal", "slain", "casualty", "trauma"},
    "violence_action": {"opened fire", "fired", "shooting", "attack", "shot", "pulled trigger",
                        "rampage", "massacre", "spree"},
    "legal_process": {"arrested", "charged", "custody", "detained", "accused", "indicted",
                      "convicted", "sentence", "trial", "court", "prosecution", "guilty"},
    "identity_age": {"year-old", "year old", "teenager", "juvenile", "minor", "adult",
                     "elderly", "young", "age", "aged"},
    "relationship": {"family", "families", "mother", "father", "parent", "son", "daughter",
                     "brother", "sister", "husband", "wife", "friend"},
    "emotion_mourning": {"mourning", "grieving", "grief", "memorial", "funeral", "tribute",
                         "devastated", "heartbroken", "tragic", "tragedy", "loss"},
}

# =============================================================================
# STOPWORDS AND PRONOUNS
# =============================================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "were", "are", "been", "be", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "need", "dare", "ought", "used", "that",
    "this", "these", "those", "it", "its", "they", "their", "them", "he", "she",
    "his", "her", "him", "we", "us", "our", "you", "your", "i", "me", "my",
    "who", "whom", "which", "what", "where", "when", "why", "how", "if", "then",
    "so", "just", "also", "only", "even", "still", "already", "yet", "again",
    "very", "too", "more", "most", "some", "any", "no", "not", "all", "each",
    "every", "both", "few", "many", "much", "other", "another", "such", "same",
}

PRONOUNS_DETERMINERS = {
    "he", "she", "it", "they", "them", "his", "her", "its", "their", "theirs",
    "him", "himself", "herself", "itself", "themselves", "this", "that", "these",
    "those", "who", "whom", "which", "what", "whose",
}

# Age pattern regex
AGE_PATTERN = re.compile(r'\b\d{1,2}[- ]?year[- ]?old\b', re.IGNORECASE)
AGE_KEYWORDS = {"teenager", "juvenile", "minor", "adult", "elderly"}


def load_spacy_model():
    """Load spaCy model for NLP processing."""
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
    return nlp


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from JSONL format."""
    filepath = Path(filepath)
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# =============================================================================
# ANCHOR DETECTION
# =============================================================================

def find_anchor_tokens(doc, entity_type: str) -> Set[int]:
    """
    Find anchor token indices for the entity type.
    
    Args:
        doc: spaCy Doc object
        entity_type: "victim" or "shooter"
        
    Returns:
        Set of anchor token indices
    """
    anchors = set()
    
    if entity_type == "shooter":
        cue_lemmas = SHOOTER_ANCHOR_LEMMAS
        cue_words = SHOOTER_CUE_WORDS
    else:
        cue_lemmas = VICTIM_ANCHOR_LEMMAS
        cue_words = VICTIM_CUE_WORDS
    
    # Find cue token positions
    cue_positions = []
    for token in doc:
        token_lower = token.text.lower()
        lemma_lower = token.lemma_.lower()
        
        if token_lower in cue_words or lemma_lower in cue_lemmas:
            anchors.add(token.i)
            cue_positions.append(token.i)
    
    # For victims: also find PERSON entities near harm terms
    if entity_type == "victim":
        harm_positions = []
        for token in doc:
            if token.lemma_.lower() in HARM_TERMS:
                harm_positions.append(token.i)
        
        # Find PERSON entities within 6 tokens of harm terms
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                for harm_pos in harm_positions:
                    if any(abs(tok.i - harm_pos) <= 6 for tok in ent):
                        for tok in ent:
                            anchors.add(tok.i)
                        break
    
    # For shooters: find PERSON entities within 6 tokens of cue words
    if entity_type == "shooter":
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                for cue_pos in cue_positions:
                    if any(abs(tok.i - cue_pos) <= 6 for tok in ent):
                        for tok in ent:
                            anchors.add(tok.i)
                        break
    
    return anchors


def subtree_contains_anchor(token, anchors: Set[int]) -> bool:
    """Check if token's subtree contains any anchor."""
    for t in token.subtree:
        if t.i in anchors:
            return True
    return False


def token_is_anchor_linked(token, anchors: Set[int]) -> bool:
    """Check if token is anchor or its subtree contains anchor."""
    return token.i in anchors or subtree_contains_anchor(token, anchors)


# =============================================================================
# ENTITY-LINKED EXTRACTION
# =============================================================================

def extract_entity_linked_predicates(doc, anchors: Set[int]) -> List[Dict]:
    """
    Extract verb predicates that are linked to anchor entities.
    
    Args:
        doc: spaCy Doc object
        anchors: Set of anchor token indices
        
    Returns:
        List of extracted predicate dictionaries
    """
    extractions = []
    
    if not anchors:
        return extractions
    
    for token in doc:
        if token.pos_ != "VERB":
            continue
        
        # Check if verb has entity-linked subject or object
        is_linked = False
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "iobj"):
                if token_is_anchor_linked(child, anchors):
                    is_linked = True
                    break
        
        # Also check if verb itself is near an anchor (within 4 tokens)
        if not is_linked:
            for anchor_idx in anchors:
                if abs(token.i - anchor_idx) <= 4:
                    is_linked = True
                    break
        
        if not is_linked:
            continue
        
        # Build predicate phrase
        predicate_tokens = [token]
        
        # Include auxiliaries
        for child in token.children:
            if child.dep_ in ("aux", "auxpass"):
                predicate_tokens.append(child)
        
        # Include negation
        for child in token.children:
            if child.dep_ == "neg":
                predicate_tokens.append(child)
        
        # Include particles (e.g., "opened fire")
        for child in token.children:
            if child.dep_ == "prt":
                predicate_tokens.append(child)
        
        # Include direct object if present
        for child in token.children:
            if child.dep_ == "dobj" and child.pos_ in ("NOUN", "PROPN"):
                predicate_tokens.append(child)
                # Include compound nouns
                for grandchild in child.children:
                    if grandchild.dep_ == "compound":
                        predicate_tokens.append(grandchild)
        
        # Include limited advmod (only high-value ones)
        for child in token.children:
            if child.dep_ == "advmod" and child.pos_ == "ADV":
                adv_lower = child.text.lower()
                if adv_lower in {"fatally", "critically", "seriously", "allegedly", "reportedly"}:
                    predicate_tokens.append(child)
        
        # Sort by position
        predicate_tokens = sorted(predicate_tokens, key=lambda t: t.i)
        
        # Max 8 tokens
        if len(predicate_tokens) > 8:
            predicate_tokens = predicate_tokens[:8]
        
        if len(predicate_tokens) >= 1:
            phrase = " ".join([t.text for t in predicate_tokens])
            start_idx = predicate_tokens[0].idx
            end_idx = predicate_tokens[-1].idx + len(predicate_tokens[-1].text)
            
            if len(phrase.strip()) > 2:
                extractions.append({
                    "description_phrase": phrase,
                    "extraction_method": "dep",
                    "token_span": f"{start_idx}-{end_idx}",
                })
    
    return extractions


def extract_entity_linked_modifiers(doc, anchors: Set[int]) -> List[Dict]:
    """
    Extract adjectival modifiers and appositives linked to anchors.
    
    Args:
        doc: spaCy Doc object
        anchors: Set of anchor token indices
        
    Returns:
        List of extracted modifier dictionaries
    """
    extractions = []
    
    if not anchors:
        return extractions
    
    for token in doc:
        # Extract adjectival modifiers (amod) only if head is anchor-linked
        if token.dep_ == "amod" and token.pos_ == "ADJ":
            head = token.head
            if head.pos_ in ("NOUN", "PROPN") and token_is_anchor_linked(head, anchors):
                phrase_tokens = [token, head]
                
                for sibling in head.children:
                    if sibling.dep_ in ("amod", "compound") and sibling != token:
                        phrase_tokens.append(sibling)
                
                phrase_tokens = sorted(phrase_tokens, key=lambda t: t.i)
                phrase = " ".join([t.text for t in phrase_tokens])
                
                if 2 <= len(phrase_tokens) <= 6:
                    extractions.append({
                        "description_phrase": phrase,
                        "extraction_method": "pos",
                        "token_span": f"{phrase_tokens[0].idx}-{phrase_tokens[-1].idx + len(phrase_tokens[-1].text)}",
                    })
        
        # Extract appositives only if attached to anchor-linked entity
        if token.dep_ == "appos":
            head = token.head
            if token_is_anchor_linked(head, anchors):
                appos_tokens = [token]
                for child in token.children:
                    if child.dep_ in ("amod", "compound", "nummod"):
                        appos_tokens.append(child)
                
                appos_tokens = sorted(appos_tokens, key=lambda t: t.i)
                phrase = " ".join([t.text for t in appos_tokens])
                
                if 1 <= len(appos_tokens) <= 6 and len(phrase) > 2:
                    extractions.append({
                        "description_phrase": phrase,
                        "extraction_method": "dep",
                        "token_span": f"{appos_tokens[0].idx}-{appos_tokens[-1].idx + len(appos_tokens[-1].text)}",
                    })
    
    return extractions


def extract_passive_predicates(doc, anchors: Set[int]) -> List[Dict]:
    """
    Extract passive constructions linked to anchors.
    """
    extractions = []
    
    for token in doc:
        if token.pos_ == "VERB" and token.tag_ == "VBN":
            has_passive_aux = any(child.dep_ == "auxpass" for child in token.children)
            
            if not has_passive_aux:
                continue
            
            # Check if linked to anchor
            is_linked = False
            for child in token.children:
                if child.dep_ == "nsubjpass" and token_is_anchor_linked(child, anchors):
                    is_linked = True
                    break
            
            if not is_linked and anchors:
                for anchor_idx in anchors:
                    if abs(token.i - anchor_idx) <= 4:
                        is_linked = True
                        break
            
            if not is_linked and anchors:
                continue
            
            phrase_tokens = [token]
            
            for child in token.children:
                if child.dep_ in ("auxpass", "aux"):
                    phrase_tokens.append(child)
            
            for child in token.children:
                if child.dep_ == "agent":
                    phrase_tokens.append(child)
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            phrase_tokens.append(grandchild)
            
            phrase_tokens = sorted(phrase_tokens, key=lambda t: t.i)
            phrase = " ".join([t.text for t in phrase_tokens])
            
            if 2 <= len(phrase_tokens) <= 8:
                extractions.append({
                    "description_phrase": phrase,
                    "extraction_method": "dep",
                    "token_span": f"{phrase_tokens[0].idx}-{phrase_tokens[-1].idx + len(phrase_tokens[-1].text)}",
                })
    
    return extractions


def is_location_only_phrase(phrase: str) -> bool:
    """Check if phrase contains only location terms without harm/violence keywords."""
    phrase_lower = phrase.lower()
    words = set(re.findall(r'\b\w+\b', phrase_lower))
    
    # Check if all content words are location-only
    content_words = words - STOPWORDS - PRONOUNS_DETERMINERS
    
    if not content_words:
        return True
    
    # If all content words are location terms
    if content_words.issubset(LOCATION_ONLY_TERMS):
        return True
    
    # Check for harm/violence keywords that would make it valuable
    harm_violence_keywords = HARM_TERMS | {"shooting", "attack", "massacre", "rampage", "gunfire"}
    if content_words & harm_violence_keywords:
        return False
    
    # If mostly location terms (>60%) and no harm terms
    location_count = len(content_words & LOCATION_ONLY_TERMS)
    if content_words and location_count / len(content_words) > 0.6:
        return True
    
    return False


def extract_noun_chunks_filtered(doc, anchors: Set[int], entity_type: str) -> List[Dict]:
    """
    Extract noun chunks that are anchor-linked or contain high-value keywords.
    Filter out location-only phrases.
    """
    extractions = []
    
    keywords = SHOOTER_KEYWORDS if entity_type == "shooter" else VICTIM_KEYWORDS
    
    for chunk in doc.noun_chunks:
        # Check if chunk contains anchor
        chunk_has_anchor = any(tok.i in anchors for tok in chunk)
        
        # Check if chunk contains high-value keyword
        chunk_lower = chunk.text.lower()
        has_keyword = any(kw in chunk_lower for kw in keywords)
        
        if not chunk_has_anchor and not has_keyword:
            continue
        
        # Filter out location-only phrases
        if is_location_only_phrase(chunk.text):
            continue
        
        phrase = chunk.text
        if 2 <= len(phrase.split()) <= 6 and len(phrase) > 3:
            extractions.append({
                "description_phrase": phrase,
                "extraction_method": "pos",
                "token_span": f"{chunk.start_char}-{chunk.end_char}",
            })
    
    return extractions


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_descriptions_from_sentence(
    sentence: str,
    nlp,
    entity_type: str
) -> List[Dict]:
    """
    Extract all descriptions from a single sentence using entity-linked extraction.
    """
    doc = nlp(sentence)
    extractions = []
    
    # Find anchor tokens for this entity type
    anchors = find_anchor_tokens(doc, entity_type)
    
    # Extract entity-linked predicates
    extractions.extend(extract_entity_linked_predicates(doc, anchors))
    
    # Extract entity-linked modifiers and appositives
    extractions.extend(extract_entity_linked_modifiers(doc, anchors))
    
    # Extract passive predicates
    extractions.extend(extract_passive_predicates(doc, anchors))
    
    # Extract filtered noun chunks
    extractions.extend(extract_noun_chunks_filtered(doc, anchors, entity_type))
    
    # Deduplicate by phrase
    seen_phrases = set()
    unique_extractions = []
    for ext in extractions:
        phrase_lower = ext["description_phrase"].lower().strip()
        if phrase_lower not in seen_phrases and len(phrase_lower) > 2:
            seen_phrases.add(phrase_lower)
            unique_extractions.append(ext)
    
    return unique_extractions


# =============================================================================
# FRAME HINT DETECTION
# =============================================================================

def detect_frame_hint(phrase: str) -> str:
    """
    Detect the frame hint for a description phrase.
    
    Args:
        phrase: The description phrase
        
    Returns:
        Frame hint string
    """
    phrase_lower = phrase.lower()
    
    # Check each frame in order of specificity
    for frame, keywords in [
        ("identity_age", FRAME_KEYWORDS["identity_age"]),
        ("legal_process", FRAME_KEYWORDS["legal_process"]),
        ("violence_action", FRAME_KEYWORDS["violence_action"]),
        ("harm", FRAME_KEYWORDS["harm"]),
        ("emotion_mourning", FRAME_KEYWORDS["emotion_mourning"]),
        ("relationship", FRAME_KEYWORDS["relationship"]),
    ]:
        for kw in keywords:
            if kw in phrase_lower:
                return frame
    
    # Check age pattern
    if AGE_PATTERN.search(phrase):
        return "identity_age"
    
    return "other"


# =============================================================================
# FILTERING
# =============================================================================

def normalize_phrase(phrase: str) -> str:
    """Normalize a phrase: strip whitespace, normalize spaces."""
    phrase = phrase.strip()
    phrase = re.sub(r'\s+', ' ', phrase)
    return phrase


def has_high_value_keyword(phrase: str, entity_type: str) -> bool:
    """Check if phrase contains a high-value keyword for the entity type."""
    phrase_lower = phrase.lower()
    keywords = SHOOTER_KEYWORDS if entity_type == "shooter" else VICTIM_KEYWORDS
    return any(keyword in phrase_lower for keyword in keywords)


def has_age_pattern(phrase: str) -> bool:
    """Check if phrase contains an age pattern."""
    if AGE_PATTERN.search(phrase):
        return True
    phrase_lower = phrase.lower()
    return any(keyword in phrase_lower for keyword in AGE_KEYWORDS)


def is_mostly_stopwords(phrase: str) -> bool:
    """Check if phrase is mostly stopwords/punctuation/numbers."""
    words = phrase.lower().split()
    if not words:
        return True
    
    content_words = [w for w in words if re.search(r'[a-zA-Z]', w)]
    if not content_words:
        return True
    
    stopword_count = sum(1 for w in content_words if w in STOPWORDS)
    return stopword_count / len(content_words) > 0.8


def is_mostly_pronouns(phrase: str) -> bool:
    """Check if phrase contains mostly pronouns/determiners."""
    words = phrase.lower().split()
    if not words:
        return True
    
    pronoun_count = sum(1 for w in words if w in PRONOUNS_DETERMINERS)
    return pronoun_count / len(words) > 0.6


def has_reporting_verb_root(phrase: str, nlp) -> bool:
    """Check if phrase's root verb is a generic reporting verb."""
    doc = nlp(phrase)
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in REPORTING_VERBS:
            if token.dep_ == "ROOT" or token.head == token:
                return True
    return False


def starts_with_infinitive(phrase: str) -> bool:
    """Check if phrase starts with infinitive 'to '."""
    return phrase.lower().startswith("to ")


def is_high_value_infinitive(phrase: str) -> bool:
    """Check if infinitive phrase contains high-value violence/legal terms."""
    high_value_terms = {
        "kill", "shoot", "murder", "attack", "wound", "injure", "arrest",
        "charge", "prosecute", "convict", "sentence", "die", "survive",
    }
    phrase_lower = phrase.lower()
    return any(term in phrase_lower for term in high_value_terms)


def filter_description(
    phrase: str,
    entity_type: str,
    nlp
) -> Tuple[bool, str]:
    """
    Apply all filtering rules to a description phrase.
    
    Returns:
        Tuple of (keep_phrase, reason)
    """
    phrase = normalize_phrase(phrase)
    tokens = phrase.split()
    
    # High-value short phrases
    high_value_short = phrase.lower() in {
        "opened fire", "shot", "killed", "wounded", "died", "dead",
        "arrested", "charged", "hospitalized", "injured",
    }
    
    if len(tokens) < 2 and not high_value_short:
        return False, "too_short"
    
    if is_mostly_stopwords(phrase):
        return False, "mostly_stopwords"
    
    if is_mostly_pronouns(phrase):
        return False, "mostly_pronouns"
    
    if has_reporting_verb_root(phrase, nlp):
        return False, "reporting_verb"
    
    if starts_with_infinitive(phrase):
        if not is_high_value_infinitive(phrase):
            return False, "generic_infinitive"
    
    # Location-only filter
    if is_location_only_phrase(phrase):
        return False, "location_only"
    
    # Require high-value keyword OR age pattern
    has_keyword = has_high_value_keyword(phrase, entity_type)
    has_age = has_age_pattern(phrase)
    
    if not has_keyword and not has_age:
        return False, "no_high_value_content"
    
    return True, "kept"


def apply_quality_filters(
    df: pd.DataFrame,
    nlp
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply quality filters to all descriptions.
    
    Returns:
        Tuple of (filtered DataFrame, filter statistics dict)
    """
    filter_stats = {
        "total_raw": len(df),
        "kept": 0,
        "too_short": 0,
        "mostly_stopwords": 0,
        "mostly_pronouns": 0,
        "reporting_verb": 0,
        "generic_infinitive": 0,
        "location_only": 0,
        "no_high_value_content": 0,
    }
    
    keep_mask = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering descriptions"):
        keep, reason = filter_description(
            row["description_phrase"],
            row["entity_type"],
            nlp
        )
        keep_mask.append(keep)
        
        # Explicitly increment stats
        if keep:
            filter_stats["kept"] += 1
        else:
            filter_stats[reason] += 1
    
    filtered_df = df[keep_mask].copy()
    filtered_df["description_phrase"] = filtered_df["description_phrase"].apply(normalize_phrase)
    
    return filtered_df, filter_stats


# =============================================================================
# PROCESSING
# =============================================================================

def process_jsonl_file(
    filepath: Path,
    entity_type: str,
    nlp
) -> List[Dict]:
    """Process a JSONL file and extract descriptions."""
    data = load_jsonl(filepath)
    records = []
    
    sentence_key = f"{entity_type}_sentences_resolved"
    
    for article in tqdm(data, desc=f"Processing {entity_type} sentences"):
        article_id = article["article_id"]
        outlet = article["outlet"]
        sentences = article.get(sentence_key, [])
        
        for sentence in sentences:
            extractions = extract_descriptions_from_sentence(sentence, nlp, entity_type)
            
            for ext in extractions:
                records.append({
                    "article_id": article_id,
                    "outlet": outlet,
                    "entity_type": entity_type,
                    "sentence_resolved": sentence,
                    "description_phrase": ext["description_phrase"],
                    "extraction_method": ext["extraction_method"],
                    "token_span": ext["token_span"],
                })
    
    return records


def generate_task2_documentation() -> str:
    """Generate documentation for Task 2."""
    doc = """# Task 2: Description Extraction Rationale

## Methods

We extract descriptive phrases using entity-linked dependency extraction:

### 1. Anchor Detection
For each sentence, we identify anchor tokens representing the target entity:
- **Shooter anchors**: Tokens matching cue words (suspect, shooter, gunman) or PERSON entities within 6 tokens
- **Victim anchors**: Tokens matching cue words (victim, student, child) or PERSON entities near harm terms

### 2. Entity-Linked Extraction
Only extract phrases grammatically linked to anchors:
- **Verb predicates**: Keep verbs whose subject/object is anchor-linked
- **Adjectival modifiers**: Keep when noun head is anchor-linked
- **Appositives**: Keep when attached to anchor entity
- **Noun chunks**: Require anchor presence OR high-value keyword

### 3. Quality Filtering
- Drop generic reporting verbs (say, tell, report)
- Drop location-only phrases (school, campus without harm terms)
- Require high-value keyword OR age pattern
- Filter stopword-heavy and pronoun-heavy phrases

### 4. Frame Hints
Each phrase tagged with semantic frame: harm, violence_action, legal_process, identity_age, relationship, emotion_mourning, other

## Strengths
1. Entity-linked extraction reduces irrelevant phrases
2. Frame hints enable semantic analysis
3. Location filtering removes noise

## Limitations
1. May miss some relevant descriptions
2. Anchor detection depends on cue word coverage

## Output
- `descriptions_raw.csv`: All extracted phrases
- `descriptions.csv`: Filtered phrases with frame_hint column
"""
    return doc


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main function to extract descriptions from contexts.
    Run with: python -m src.extract_descriptions
    """
    print("=" * 60)
    print("Description Extraction (Entity-Linked + Quality Filtering)")
    print("=" * 60)
    
    config.ensure_output_dirs()
    
    victim_path = config.PROCESSED_DIR / "contexts_victims.jsonl"
    shooter_path = config.PROCESSED_DIR / "contexts_shooters.jsonl"
    
    if not victim_path.exists() or not shooter_path.exists():
        print("Error: Context files not found.")
        print("Please run 'python -m src.coref_contexts' first.")
        return
    
    print("\nLoading spaCy model...")
    nlp = load_spacy_model()
    
    # Process contexts
    print("\nProcessing victim contexts...")
    victim_records = process_jsonl_file(victim_path, "victim", nlp)
    print(f"Extracted {len(victim_records)} raw victim descriptions")
    
    print("\nProcessing shooter contexts...")
    shooter_records = process_jsonl_file(shooter_path, "shooter", nlp)
    print(f"Extracted {len(shooter_records)} raw shooter descriptions")
    
    # Combine raw records
    all_records = victim_records + shooter_records
    raw_df = pd.DataFrame(all_records)
    
    column_order = [
        "article_id", "outlet", "entity_type", "sentence_resolved",
        "description_phrase", "extraction_method", "token_span"
    ]
    raw_df = raw_df[column_order]
    
    # Save raw descriptions
    raw_output_path = config.PROCESSED_DIR / "descriptions_raw.csv"
    utils_io.save_dataframe(raw_df, raw_output_path)
    print(f"\nSaved RAW descriptions to: {raw_output_path}")
    
    # Apply quality filters
    print("\nApplying quality filters...")
    filtered_df, filter_stats = apply_quality_filters(raw_df, nlp)
    
    # Add frame_hint column
    print("\nDetecting frame hints...")
    filtered_df["frame_hint"] = filtered_df["description_phrase"].apply(detect_frame_hint)
    
    # Reorder columns (frame_hint at end for backward compatibility)
    final_columns = [
        "article_id", "outlet", "entity_type", "sentence_resolved",
        "description_phrase", "extraction_method", "token_span", "frame_hint"
    ]
    filtered_df = filtered_df[final_columns]
    
    # Save filtered descriptions
    output_path = config.PROCESSED_DIR / "descriptions.csv"
    utils_io.save_dataframe(filtered_df, output_path)
    print(f"Saved FILTERED descriptions to: {output_path}")
    
    # Print filter statistics
    print("\n" + "=" * 60)
    print("FILTERING STATISTICS")
    print("=" * 60)
    
    print(f"\nRaw count:      {filter_stats['total_raw']}")
    print(f"Filtered count: {filter_stats['kept']}")
    if filter_stats['total_raw'] > 0:
        print(f"Removal rate:   {(1 - filter_stats['kept']/filter_stats['total_raw'])*100:.1f}%")
    
    print("\nDropped by reason:")
    for reason in ["too_short", "mostly_stopwords", "mostly_pronouns", 
                   "reporting_verb", "generic_infinitive", "location_only", 
                   "no_high_value_content"]:
        count = filter_stats.get(reason, 0)
        if count > 0:
            pct = count / filter_stats['total_raw'] * 100 if filter_stats['total_raw'] > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Statistics on filtered data
    print("\n" + "=" * 60)
    print("FILTERED DATA STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal descriptions: {len(filtered_df)}")
    
    print("\nBy entity type:")
    print(filtered_df.groupby("entity_type").size().to_string())
    
    print("\nBy extraction method:")
    print(filtered_df.groupby("extraction_method").size().to_string())
    
    print("\nBy outlet:")
    print(filtered_df.groupby("outlet").size().to_string())
    
    print("\nBy frame_hint:")
    print(filtered_df.groupby("frame_hint").size().sort_values(ascending=False).to_string())
    
    # Top 20 phrases by entity type
    print("\n" + "=" * 60)
    print("TOP 20 PHRASES BY ENTITY TYPE")
    print("=" * 60)
    
    for entity_type in ["victim", "shooter"]:
        entity_df = filtered_df[filtered_df["entity_type"] == entity_type]
        print(f"\n{entity_type.upper()} (top 20):")
        phrase_counts = entity_df["description_phrase"].str.lower().value_counts().head(20)
        for i, (phrase, count) in enumerate(phrase_counts.items(), 1):
            print(f"  {i:2}. {phrase} ({count})")
    
    # Save documentation
    doc_path = config.REPORTS_DIR / "task2_rationale.md"
    doc_content = generate_task2_documentation()
    utils_io.write_text_file(doc_path, doc_content)
    print(f"\nDocumentation saved to: {doc_path}")
    
    # Sample extractions
    print("\n" + "=" * 60)
    print("SAMPLE FILTERED EXTRACTIONS")
    print("=" * 60)
    
    print("\nVictim descriptions (sample):")
    victim_sample = filtered_df[filtered_df["entity_type"] == "victim"].head(10)
    for _, row in victim_sample.iterrows():
        print(f"  [{row['extraction_method']}|{row['frame_hint']}] {row['description_phrase']}")
    
    print("\nShooter descriptions (sample):")
    shooter_sample = filtered_df[filtered_df["entity_type"] == "shooter"].head(10)
    for _, row in shooter_sample.iterrows():
        print(f"  [{row['extraction_method']}|{row['frame_hint']}] {row['description_phrase']}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
