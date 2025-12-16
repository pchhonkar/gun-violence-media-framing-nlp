# Task 1: Coreference Resolution and Context Extraction

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
