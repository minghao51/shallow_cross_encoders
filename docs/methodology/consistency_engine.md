# Consistency Engine

## Overview

A contradiction detection system that extracts structured claims from documents using regex patterns, groups them by entity and attribute, and identifies conflicting values. It operates in two modes: exact attribute matching (fast path) and semantic attribute matching (fuzzy path) using embedding similarity.

**Implementation**: `src/reranker/strategies/consistency.py`

**Pattern definitions**: `src/reranker/strategies/patterns.py` (22 regex patterns)

---

## Mathematical Formulation

### Claim Extraction

Each document D is parsed into a set of structured claims:

```
Claim = (entity, attribute, value, source_doc_id)
```

Claims are extracted by matching 22 regex patterns against document segments. Each pattern captures three named groups:
- `entity`: The subject entity (e.g., "Python 3.12", "GPT-4")
- `attribute`: The property being stated (e.g., "release_date", "context_window")
- `value`: The stated value (e.g., "October 2023", "128K tokens")

### Normalization Functions

Three normalization functions clean extracted claims:

**Entity normalization**:
```
normalize_entity(e) = strip_narrative_prefixes(e) → strip_copula_suffixes(e)
```
Removes narrative prefixes like "the", "however", "in fact", "contrary to popular belief" and copula suffixes like "is ...", "has ...", "was ...".

**Attribute normalization**:
```
normalize_attribute(a) = lowercase(a) → replace_spaces_with_underscores(a) → strip_articles(a) → apply_domain_rules(a)
```
Maps semantically equivalent attributes to canonical forms (e.g., "latency" variants → "latency").

**Value normalization**:
```
normalize_value(v) = strip_reported_prefixes(v) → strip_approximation_markers(v) → strip_context_suffixes(v)
```
Removes hedging language like "reported to be", "approximately", "only".

### Contradiction Detection

#### Phase 1: Exact Attribute Matching (Fast Path)

Claims are grouped by (entity, attribute). Within each group, pairs from different documents are compared:

```
contradiction(cᵢ, cⱼ) = (entityᵢ = entityⱼ) ∧ (attributeᵢ = attributeⱼ) ∧ (sourceᵢ ≠ sourceⱼ) ∧ values_conflict(valueᵢ, valueⱼ)
```

**Value conflict function**:
```
values_conflict(v₁, v₂) = { |float(v₁) - float(v₂)| > tolerance   if both numeric
                          { str(v₁).lower() ≠ str(v₂).lower()     otherwise
```

#### Phase 2: Semantic Attribute Matching (Fuzzy Path)

For claims with different but semantically similar attributes, embedding similarity is used:

```
similar_attributes(aᵢ, aⱼ) = 1 - cosine_distance(embed(aᵢ), embed(aⱼ)) ≥ sim_threshold
```

Where `sim_threshold` defaults to 0.95 (cosine distance ≤ 0.05).

```
contradiction_fuzzy(cᵢ, cⱼ) = (entityᵢ = entityⱼ) ∧ similar_attributes(aᵢ, aⱼ) ∧ (sourceᵢ ≠ sourceⱼ) ∧ values_conflict(vᵢ, vⱼ)
```

#### Phase 3: Fuzzy Claim Matching

For claims that couldn't be parsed structurally (stored as `raw_text`), full claim embeddings are compared:

```
similar_claims(cᵢ, cⱼ) = 1 - cosine_distance(embed(entityᵢ + " " + attributeᵢ), embed(entityⱼ + " " + attributeⱼ)) ≥ sim_threshold
```

### Contradiction Scoring

Documents are ranked by their contradiction penalty:

```
penalty(doc) = count(contradictions involving doc)
score(doc) = -penalty(doc)
```

Documents with fewer contradictions are ranked higher (lower penalty = higher score).

---

## DAG Components

```
┌─────────────────────────────────────────────────────────────┐
│                  CLAIM EXTRACTION                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              Documents                      │
│  [doc_0, doc_1, ..., doc_n]                 │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Sentence Splitting                   │
│  Split on [.!?] ;                           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Pattern Hint Pre-filtering           │
│  For each segment, check hint substrings    │
│  Only try patterns whose hints match        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Regex Pattern Matching               │
│  22 structured patterns                     │
│  Capture: entity, attribute, value          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Claim Normalization                  │
│  normalize_entity()                         │
│  normalize_attribute()                      │
│  normalize_value()                          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        ClaimSet (per document)              │
│  [Claim(entity, attr, value, doc_id), ...]  │
└─────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│              CONTRADICTION DETECTION                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│        All Claims from All Documents        │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  Structured      │  │  Fuzzy (raw_text)    │
│  Claims          │  │  Claims              │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Group by entity (normalized)                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  Phase 1: Exact  │  │  Phase 2: Semantic   │
│  Attribute Match │  │  Attribute Match     │
│                  │  │                      │
│  Group by attr   │  │  Embed attributes    │
│  Compare values  │  │  Compute cosine dist │
│  within group    │  │  If dist < threshold │
└────────┬─────────┘  │  Compare values      │
         │            └──────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Fuzzy Claim Matching                              │
│  Embed (entity + attribute) for raw_text claims             │
│  Compare pairwise if cosine dist < threshold                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Contradiction List                   │
│  [Contradiction(claim_a, claim_b, reason)]  │
└─────────────────────────────────────────────┘
```

---

## Approach & Methodology

### Claim Extraction

1. **Split documents** into sentences (on `.`, `!`, `?`, `;`)
2. **For each sentence segment**:
   - Pre-filter patterns using hint substrings (fast rejection)
   - Try matching candidate patterns (or all patterns if no hints match)
   - Extract entity, attribute, value from named capture groups
3. **Normalize** each extracted claim (entity, attribute, value)
4. **Deduplicate** claims by (entity, attribute, value) key
5. **Fallback**: If no structured claims found, store entire document as raw_text claim

### Contradiction Detection

1. **Separate claims** into structured and fuzzy (raw_text) groups
2. **Group structured claims** by normalized entity
3. **For each entity group**:
   - **Fast path**: Group by exact attribute, compare values within same-attribute groups
   - **Fuzzy path**: Embed unique attributes, compute pairwise cosine distances, compare values for semantically similar attributes
4. **Fuzzy claim matching**: Embed (entity + attribute) for raw_text claims, compare pairwise
5. **Value conflict check**: Numeric comparison with tolerance, or string equality

### Reranking

1. **Extract claims** from all documents
2. **Detect contradictions** across all claim pairs
3. **Count penalties**: Each contradiction adds 1.0 to both involved documents
4. **Rank by penalty**: Documents with fewer contradictions rank higher

### Pattern Architecture

**22 regex patterns** organized with hint-based pre-filtering:

| Pattern | Example Match | Hints |
|---------|--------------|-------|
| reports/lists/states/shows | "X reports Y as Z" | " reports ", " shows " |
| confirms (no connector) | "X confirms Y Z" | " confirms " |
| release date | "X will be released in 2024" | " release in " |
| screening status | "screening status of X is Y" | "screening status" |
| best metric | "best metric for X is Y" | "best metric" |
| possessive | "X's Y is Z" | "'s " |
| has a/of | "X has a Y of Z" | " has ", " of " |
| boasts/achieves | "X boasts a Y of Z" | " boasts " |
| shows/operates with | "X shows a Y of Z" | " shows ", " exhibits " |
| recorded | "X recorded a Y of Z" | " recorded " |
| known for | "X is known for its Y of Z" | " known for " |
| for/within | "for X, Y is Z" | "for ", "within " |
| the Y of X is | "the Y of X is Z" | " of ", " is " |
| measured/reported | "the Y of X is measured at Z" | " measured at " |
| reported/increased | "the Y for X has increased to Z" | " for ", " increased to " |
| has been reported | "X has been reported to have a Y of Z" | " reported to have " |
| actually | "the Y of X is actually Z" | " actually " |
| for (variant) | "Y for X is Z" | " for " |
| with Y of Z, X | "with a Y of Z, X ..." | "with ", " of " |
| colon/equals | "X: Y = Z" | ":", "=", " - " |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Regex-based extraction** | Fast, deterministic, no LLM dependency |
| **Hint pre-filtering** | Avoids trying all 22 patterns on every sentence |
| **Three-phase detection** | Exact match first (fastest), then semantic, then fuzzy |
| **Embedding-based attribute similarity** | Handles paraphrased attributes (e.g., "release_date" vs "launch_date") |
| **Numeric tolerance** | Handles floating-point differences (default tolerance: 0.01) |

### Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `sim_threshold` | 0.95 | Minimum cosine similarity for attribute matching |
| `value_tolerance` | 0.01 | Maximum numeric difference before values conflict |

### Performance

| Metric | Value |
|--------|-------|
| Latency | ~0.07ms |
| Recall (expanded v2) | 100.0% |
| False Positive Rate | 0.0% |
| Contradiction subjects | 35 |
| Total contradictions | 990 |
| Control pairs | 400 |

### When to Use

- **Fact-checking**: Detect conflicting claims across multiple sources
- **Multi-document QA**: Identify which documents agree/disagree on facts
- **Data quality**: Find inconsistencies in structured data
- **Reranking**: Penalize documents that contradict other sources

### Limitations

- **Pattern coverage**: Only extracts claims matching predefined patterns
- **Narrative text**: Struggles with unstructured, narrative prose
- **Cross-entity contradictions**: Only detects contradictions about the same entity
- **Numeric parsing**: Requires values to be parseable as floats for numeric comparison
