# Task 2: Description Extraction Rationale

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
