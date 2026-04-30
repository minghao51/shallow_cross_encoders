# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=1.26.0",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", app_title="Consistency Engine Demo")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo

    return Path, json, mo


@app.cell
def _(mo):
    mo.md("""
    # Consistency Engine Demo

    Detect **factual contradictions** across documents using the
    `ConsistencyEngine`. Enter documents containing structured claims
    and see which ones contradict each other — all running on CPU.
    """)
    return


@app.cell
def _():
    from reranker.embedder import Embedder
    from reranker.strategies.consistency import ConsistencyEngine

    return ConsistencyEngine, Embedder


@app.cell
def _(ConsistencyEngine, Embedder):
    engine = ConsistencyEngine(embedder=Embedder())
    engine
    return (engine,)


@app.cell
def _(Path, json):
    contradictions_path = Path("data/raw/contradictions.jsonl")
    contradiction_data = []
    if contradictions_path.exists():
        with open(contradictions_path) as f:
            contradiction_data = [json.loads(line) for line in f if line.strip()]
    contradiction_data
    return (contradiction_data,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Sample Data or Enter Custom Documents

    Use the pre-generated contradiction dataset, or type your own documents below.
    """)
    return


@app.cell
def _(contradiction_data, mo):
    sample_docs_a = (
        [c["doc_a"] for c in contradiction_data[:8]]
        if contradiction_data
        else [
            "Model2Vec Potion-8M reports latency_ms as 2.",
            "HybridFusionReranker reports best_metric as NDCG@10.",
        ]
    )
    sample_docs_b = (
        [c["doc_b"] for c in contradiction_data[:8]]
        if contradiction_data
        else [
            "Model2Vec Potion-8M reports latency_ms as 7.",
            "HybridFusionReranker reports best_metric as MRR.",
        ]
    )

    use_sample_toggle = mo.ui.radio(
        options=["Use sample data", "Enter custom documents"],
        value="Use sample data",
        label="Data source",
    )

    custom_docs = mo.ui.text_area(
        value="ModelX reports latency as 50ms.\nModelX reports latency as 12ms.\nModelY reports accuracy as 0.95.",
        label="Custom documents (one per line)",
        full_width=True,
        rows=5,
    )

    sim_threshold = mo.ui.slider(
        start=0.8,
        stop=1.0,
        step=0.01,
        value=0.95,
        label="Similarity threshold (higher = stricter matching)",
    )

    value_tolerance = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.01,
        label="Value tolerance (for numeric comparisons)",
    )

    use_sample_toggle, custom_docs, sim_threshold, value_tolerance
    return (
        custom_docs,
        sample_docs_a,
        sample_docs_b,
        sim_threshold,
        use_sample_toggle,
        value_tolerance,
    )


@app.cell
def _(custom_docs, sample_docs_a, sample_docs_b, use_sample_toggle):
    if use_sample_toggle.value == "Use sample data":
        docs = sample_docs_a + sample_docs_b
    else:
        docs = [d.strip() for d in custom_docs.value.strip().split("\n") if d.strip()]
    docs
    return (docs,)


@app.cell
def _(docs, engine, sim_threshold, value_tolerance):
    engine.sim_threshold = sim_threshold.value
    engine.value_tolerance = value_tolerance.value

    extracted_claims = engine.extract_claims(docs)
    detected_contradictions = engine.check(extracted_claims)
    extracted_claims, detected_contradictions
    return detected_contradictions, extracted_claims


@app.cell
def _(docs, extracted_claims, mo):
    mo.md("## 2. Extracted Claims")
    claim_sections = []
    for idx, (cs, doc_text) in enumerate(zip(extracted_claims, docs)):
        rows = []
        for claim in cs.claims:
            rows.append(
                f"| {claim.entity} | {claim.attribute} | `{claim.value}` | {claim.source_doc_id} |"
            )
        if rows:
            table = (
                "| Entity | Attribute | Value | Source |\n"
                "|--------|-----------|-------|--------|\n" + "\n".join(rows)
            )
        else:
            table = "*No structured claims extracted.*"
        claim_sections.append(mo.md(f"### Document {idx + 1}\n> {doc_text[:150]}\n\n{table}"))
    mo.vstack(claim_sections)
    return


@app.cell
def _(detected_contradictions, mo):
    mo.md(
        f"## 3. Detected Contradictions\n\nFound **{len(detected_contradictions)}** contradictions."
    )
    return


@app.cell
def _(detected_contradictions, mo):
    sections = []
    for i, c in enumerate(detected_contradictions):
        sections.append(
            mo.md(
                f"### Contradiction {i + 1}\n"
                f"- **Claim A:** `{c.claim_a.entity}` → {c.claim_a.attribute} = `{c.claim_a.value}` "
                f"(from {c.claim_a.source_doc_id})\n"
                f"- **Claim B:** `{c.claim_b.entity}` → {c.claim_b.attribute} = `{c.claim_b.value}` "
                f"(from {c.claim_b.source_doc_id})\n"
                f"- **Reason:** {c.reason}\n"
            )
        )
    if not sections:
        sections.append(
            mo.md(
                "*No contradictions detected. Try adjusting the thresholds or adding documents with conflicting values.*"
            )
        )
    mo.vstack(sections)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Batch Evaluation on Labeled Data
    """)
    return


@app.cell
def _(contradiction_data, engine, mo):
    if not contradiction_data:
        eval_result = mo.md("*No labeled contradiction data available for evaluation.*")
    else:
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for item in contradiction_data:
            expected = item.get("is_contradiction", True)
            eval_claims = engine.extract_claims([item["doc_a"], item["doc_b"]], ["doc_a", "doc_b"])
            detected = len(engine.check(eval_claims)) > 0
            if expected and detected:
                tp += 1
            elif expected and not detected:
                fn += 1
            elif not expected and detected:
                fp += 1
            else:
                tn += 1

        total = tp + fp + fn + tn
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        eval_result = mo.md(
            f"### Evaluation Results\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total samples | {total} |\n"
            f"| True Positives | {tp} |\n"
            f"| False Positives | {fp} |\n"
            f"| False Negatives | {fn} |\n"
            f"| True Negatives | {tn} |\n"
            f"| **Recall** | **{recall:.2%}** |\n"
            f"| **Precision** | **{precision:.2%}** |\n"
            f"| **F1** | **{f1:.2%}** |"
        )
    eval_result
    return


if __name__ == "__main__":
    app.run()
