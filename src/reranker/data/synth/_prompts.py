"""Prompt templates for teacher-based synthetic data generation."""

from __future__ import annotations

PAIR_PROMPT = """
You are generating training data for a lightweight reranking system.
Produce one query-document pair with a graded relevance label.

Return ONLY valid JSON with this schema:
{{
  "query": "<search query>",
  "doc": "<candidate document>",
  "score": 0 | 1 | 2 | 3,
  "rationale": "<one concise sentence>"
}}

Label meaning:
0 = irrelevant
1 = tangential
2 = relevant
3 = highly relevant

Use this seed topic for inspiration:
Query seed: {query}
Relevant seed snippet: {positive}
Irrelevant seed snippet: {negative}
Preferred label for this sample: {target_score}
"""

PAIR_BATCH_PROMPT = """
You are generating training data for a lightweight reranking system.
Return ONLY valid JSON with this schema:
{{
  "records": [
    {{
      "query": "<search query>",
      "doc": "<candidate document>",
      "score": 0 | 1 | 2 | 3,
      "rationale": "<one concise sentence>"
    }}
  ]
}}

Requirements:
- Return exactly {count} records.
- Respect the requested seed topic and target score for each item.
- Keep outputs concise and diverse.

Seed items:
{items_json}
"""

PREFERENCE_PROMPT = """
You are generating training data for a pairwise reranking model.
Produce one preference example for the query below.

Return ONLY valid JSON with this schema:
{{
  "query": "<search query>",
  "doc_a": "<candidate document A>",
  "doc_b": "<candidate document B>",
  "preferred": "A" | "B",
  "confidence": <float between 0.0 and 1.0>
}}

Requirements:
- The preferred document should clearly answer the query better.
- Keep both documents plausible and topically related when possible.
- Avoid ties.

Use this seed topic for inspiration:
Query seed: {query}
Relevant seed snippet: {positive}
Irrelevant seed snippet: {negative}
"""

PREFERENCE_BATCH_PROMPT = """
You are generating training data for a pairwise reranking model.
Return ONLY valid JSON with this schema:
{{
  "records": [
    {{
      "query": "<search query>",
      "doc_a": "<candidate document A>",
      "doc_b": "<candidate document B>",
      "preferred": "A" | "B",
      "confidence": <float between 0.0 and 1.0>
    }}
  ]
}}

Requirements:
- Return exactly {count} records.
- Keep both documents plausible and topically related.
- Avoid ties.

Seed items:
{items_json}
"""

CONTRADICTION_PROMPT = """
Generate two short document excerpts about the same subject.

Return ONLY valid JSON with this schema:
{{
  "subject": "<entity name>",
  "doc_a": "<text>",
  "doc_b": "<text>",
  "contradicted_field": "<field name>",
  "value_a": "<value stated in doc_a>",
  "value_b": "<value stated in doc_b>",
  "is_contradiction": <true or false>
}}

Requirements:
- Keep the subject and surrounding context similar across both documents.
- If is_contradiction is true, doc_b must contradict doc_a on the specified field.
- If is_contradiction is false, both documents should agree on the specified field.

Preferred setting:
Subject seed: {subject}
Field seed: {field}
Value A seed: {value_a}
Value B seed: {value_b}
Target contradiction label: {target_label}
"""

CONTRADICTION_BATCH_PROMPT = """
Generate contradiction-training examples.
Return ONLY valid JSON with this schema:
{{
  "records": [
    {{
      "subject": "<entity name>",
      "doc_a": "<text>",
      "doc_b": "<text>",
      "contradicted_field": "<field name>",
      "value_a": "<value stated in doc_a>",
      "value_b": "<value stated in doc_b>",
      "is_contradiction": <true or false>
    }}
  ]
}}

Requirements:
- Return exactly {count} records.
- Keep the subject and surrounding context similar across paired documents.
- Respect the requested target contradiction label for each seed item.

Seed items:
{items_json}
"""

HARD_NEGATIVE_PROMPT = """
You are generating hard negative examples for a reranking system.
Given a query and a positive (relevant) document, produce:
1. A hard negative: semantically similar to the query but NOT actually answering it
2. An easy negative: obviously irrelevant to the query

Return ONLY valid JSON with this schema:
{{
  "query": "<search query>",
  "positive": "<relevant document>",
  "hard_negative": "<semantically similar but wrong answer>",
  "easy_negative": "<obviously unrelated text>",
  "hard_negative_reason": "<why this looks relevant but isn't>"
}}

Query: {query}
Positive document: {positive}
"""

HARD_NEGATIVE_BATCH_PROMPT = """
You are generating hard negative examples for a reranking system.
Return ONLY valid JSON with this schema:
{{
  "records": [
    {{
      "query": "<search query>",
      "positive": "<relevant document>",
      "hard_negative": "<semantically similar but wrong answer>",
      "easy_negative": "<obviously unrelated text>",
      "hard_negative_reason": "<why this looks relevant but isn't>"
    }}
  ]
}}

Requirements:
- Return exactly {count} records.
- Hard negatives should share vocabulary/topics with the query but fail to answer it.
- Easy negatives should be completely off-topic.

Seed items:
{items_json}
"""

LISTWISE_PROMPT = """
You are generating listwise preference data for a reranking system.
Given a query and a pool of documents, rank them by relevance and assign scores.

Return ONLY valid JSON with this schema:
{{
  "query": "<search query>",
  "docs": ["<doc1>", "<doc2>", "<doc3>", "<doc4>"],
  "scores": [0.9, 0.6, 0.3, 0.1]
}}

Requirements:
- Scores should be between 0.0 and 1.0, normalized (sum to ~1.0 is fine).
- Higher score = more relevant to the query.
- Ensure clear differentiation between relevance levels.

Query: {query}
Document pool:
{docs_json}
"""

LISTWISE_BATCH_PROMPT = """
You are generating listwise preference data for a reranking system.
Return ONLY valid JSON with this schema:
{{
  "records": [
    {{
      "query": "<search query>",
      "docs": ["<doc1>", "<doc2>", ...],
      "scores": [<float>, <float>, ...]
    }}
  ]
}}

Requirements:
- Return exactly {count} records.
- Each record should have 3-5 docs ranked by relevance.
- Scores between 0.0 and 1.0, higher = more relevant.

Seed items:
{items_json}
"""

QUERY_EXPANSION_PROMPT = """
You are generating alternative query phrasings for a search system.
Given an original query, produce 3-5 alternative ways to express the same intent.

Return ONLY valid JSON with this schema:
{{
  "original_query": "<original search query>",
  "expanded_queries": [
    "<alternative phrasing 1>",
    "<alternative phrasing 2>",
    "<alternative phrasing 3>"
  ]
}}

Requirements:
- All expanded queries should preserve the original intent.
- Vary the wording, specificity, and structure.
- Include both shorter and longer variants.

Original query: {query}
"""

QUERY_EXPANSION_BATCH_PROMPT = """
You are generating alternative query phrasings for a search system.
Return ONLY valid JSON with this schema:
{{
  "records": [
    {{
      "original_query": "<original search query>",
      "expanded_queries": ["<alt1>", "<alt2>", "<alt3>"]
    }}
  ]
}}

Requirements:
- Return exactly {count} records.
- Each record should have 3-5 expanded queries.
- Preserve original intent while varying phrasing.

Seed items:
{items_json}
"""
