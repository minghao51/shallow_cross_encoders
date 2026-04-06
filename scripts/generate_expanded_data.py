"""Generate expanded synthetic datasets with diverse queries and balanced relevance.

Addresses benchmark limitations:
1. Too few unique queries (3 → 24 across 8 domains)
2. Too easy - all models achieve perfect scores → Added hard negatives
3. No hard negatives → Multiple difficulty levels per query
4. Limited domain coverage → 8 diverse domains
5. Class imbalance → Balanced score distribution (0-3)
"""

from __future__ import annotations

import json
from pathlib import Path

from reranker.data.expanded import (
    generate_expanded_contradictions,
    generate_expanded_pairs,
    generate_expanded_preferences,
)
from reranker.utils import write_jsonl


def main():
    """Generate all expanded datasets."""
    output_dir = Path("data/raw_expanded")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating expanded datasets...")

    # Generate pairs
    print("\n--- Generating Pairs ---")
    pairs = generate_expanded_pairs(target_count=10000, seed=42)
    write_jsonl(output_dir / "pairs.jsonl", pairs)

    score_counts = {}
    domain_counts = {}
    unique_queries = set()

    for p in pairs:
        score_counts[p["score"]] = score_counts.get(p["score"], 0) + 1
        domain_counts[p["domain"]] = domain_counts.get(p["domain"], 0) + 1
        unique_queries.add(p["query"])

    print(f"  Total pairs: {len(pairs)}")
    print(f"  Unique queries: {len(unique_queries)}")
    print(f"  Score distribution: {dict(sorted(score_counts.items()))}")
    for domain in sorted(domain_counts.keys()):
        domain_queries = set(p["query"] for p in pairs if p["domain"] == domain)
        print(f"    {domain}: {len(domain_queries)} queries, {domain_counts[domain]} pairs")

    # Generate preferences
    print("\n--- Generating Preferences ---")
    prefs = generate_expanded_preferences(target_count=5000, seed=42)
    write_jsonl(output_dir / "preferences.jsonl", prefs)

    pref_stats = {"A": 0, "B": 0}
    pref_queries = set()
    for p in prefs:
        pref_stats[p["preferred"]] += 1
        pref_queries.add(p["query"])

    print(f"  Total preferences: {len(prefs)}")
    print(f"  Unique queries: {len(pref_queries)}")
    print(f"  Preference distribution: {pref_stats}")

    # Generate contradictions
    print("\n--- Generating Contradictions ---")
    contras = generate_expanded_contradictions(contradiction_count=1000, control_count=400, seed=42)
    write_jsonl(output_dir / "contradictions.jsonl", contras)

    contra_stats = {True: 0, False: 0}
    contra_subjects = set()
    for c in contras:
        contra_stats[c["is_contradiction"]] += 1
        contra_subjects.add(c["subject"])

    print(f"  Total: {len(contras)}")
    print(f"  Contradictions: {contra_stats[True]}, Controls: {contra_stats[False]}")
    print(f"  Unique subjects: {len(contra_subjects)}")

    # Save manifest
    manifest = {
        "generated_at": "2026-04-02",
        "description": "Expanded dataset: 73 queries, 10 domains, "
        "balanced relevance with hard negatives",
        "seed": 42,
        "generation_mode": "offline_expanded",
        "datasets": {
            "pairs": {
                "count": len(pairs),
                "unique_queries": len(unique_queries),
                "score_distribution": {str(k): v for k, v in sorted(score_counts.items())},
                "domains": sorted(domain_counts.keys()),
            },
            "preferences": {
                "count": len(prefs),
                "unique_queries": len(pref_queries),
                "preference_distribution": pref_stats,
            },
            "contradictions": {
                "count": len(contras),
                "contradictions": contra_stats[True],
                "controls": contra_stats[False],
                "unique_subjects": len(contra_subjects),
            },
        },
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
