# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=1.26.0",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", app_title="Interactive Reranking Explorer")


@app.cell
def _():
    import json
    import time
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, time


@app.cell
def _(mo):
    mo.md("""
    # Interactive Reranking Explorer

    Compare **Hybrid Fusion**, **Distilled Pairwise**, and **Shallow ColBERT**
    reranking strategies side-by-side on your own queries and documents.

    All models run **CPU-only** — no GPU or API calls needed.
    """)
    return


@app.cell
def _():
    from reranker.embedder import Embedder
    from reranker.strategies.distilled import DistilledPairwiseRanker
    from reranker.strategies.hybrid import HybridFusionReranker
    from reranker.strategies.late_interaction import StaticColBERTReranker

    return (
        DistilledPairwiseRanker,
        Embedder,
        HybridFusionReranker,
        StaticColBERTReranker,
    )


@app.cell
def _(Path, json, mo):
    pairs_path = Path("data/raw/pairs.jsonl")
    preferences_path = Path("data/raw/preferences.jsonl")

    pairs_data = []
    if pairs_path.exists():
        with open(pairs_path) as f:
            pairs_data = [json.loads(line) for line in f if line.strip()]

    preferences_data = []
    if preferences_path.exists():
        with open(preferences_path) as f:
            preferences_data = [json.loads(line) for line in f if line.strip()]

    unique_queries = list(dict.fromkeys(p["query"] for p in pairs_data)) if pairs_data else []
    unique_docs = list(dict.fromkeys(p["doc"] for p in pairs_data)) if pairs_data else []

    data_status = mo.md(
        f"**Loaded:** {len(pairs_data)} pairs, {len(preferences_data)} preferences, "
        f"{len(unique_queries)} unique queries, {len(unique_docs)} unique docs"
    )
    data_status
    return pairs_data, preferences_data, unique_docs, unique_queries


@app.cell
def _(Embedder, mo):
    embedder = Embedder()
    backend_info = embedder.describe()
    dim = embedder.dimension

    embedder_md = mo.md(
        f"**Embedder:** `{backend_info['backend']}` | "
        f"Model: `{backend_info['model_name']}` | "
        f"Dimension: `{dim}`"
    )
    embedder_md
    return (embedder,)


@app.cell
def _(
    DistilledPairwiseRanker,
    HybridFusionReranker,
    StaticColBERTReranker,
    embedder,
    mo,
    pairs_data,
    preferences_data,
):
    def train_hybrid(pairs, emb):
        reranker = HybridFusionReranker(embedder=emb)
        queries = [p["query"] for p in pairs]
        docs = [p["doc"] for p in pairs]
        scores = [float(p["score"]) for p in pairs]
        reranker.fit_pointwise(queries, docs, scores, use_regression=True)
        return reranker

    def train_distilled(prefs, emb):
        reranker = DistilledPairwiseRanker(embedder=emb)
        queries = [p["query"] for p in prefs]
        doc_as = [p["doc_a"] for p in prefs]
        doc_bs = [p["doc_b"] for p in prefs]
        labels = [1 if p["preferred"] == "A" else 0 for p in prefs]
        reranker.fit(queries, doc_as, doc_bs, labels)
        return reranker

    def build_colbert(docs, emb):
        reranker = StaticColBERTReranker(embedder=emb)
        reranker.fit(docs)
        return reranker

    hybrid_reranker = train_hybrid(pairs_data, embedder) if pairs_data else None
    distilled_reranker = train_distilled(preferences_data, embedder) if preferences_data else None
    colbert_reranker = (
        build_colbert(list(set(p["doc"] for p in pairs_data)), embedder) if pairs_data else None
    )

    trained_md = mo.md(
        f"**Models trained:** "
        f"Hybrid={'yes' if hybrid_reranker and hybrid_reranker.is_fitted else 'no'}, "
        f"Distilled={'yes' if distilled_reranker and distilled_reranker.is_fitted else 'no'}, "
        f"ColBERT={'yes' if colbert_reranker and colbert_reranker.is_fitted else 'no'}"
    )
    trained_md
    return colbert_reranker, distilled_reranker, hybrid_reranker


@app.cell
def _(mo, unique_docs, unique_queries):
    query_input = mo.ui.text(
        value=unique_queries[0] if unique_queries else "python dataclass default factory",
        label="Query",
        full_width=True,
    )

    docs_input = mo.ui.text_area(
        value="\n".join(unique_docs[:10]) if unique_docs else "Enter documents, one per line",
        label="Documents (one per line)",
        full_width=True,
        rows=8,
    )

    strategy_checkboxes = mo.ui.dictionary(
        {
            "Hybrid Fusion": mo.ui.checkbox(value=True),
            "Distilled Pairwise": mo.ui.checkbox(value=True),
            "Shallow ColBERT": mo.ui.checkbox(value=True),
        }
    )

    mo.vstack([query_input, docs_input, mo.md("**Strategies:**"), strategy_checkboxes])
    return docs_input, query_input, strategy_checkboxes


@app.cell
def _(docs_input, query_input):
    active_query = query_input.value.strip()
    active_docs = [d.strip() for d in docs_input.value.strip().split("\n") if d.strip()]
    active_query, active_docs
    return active_docs, active_query


@app.cell
def _(
    active_docs,
    active_query,
    colbert_reranker,
    distilled_reranker,
    hybrid_reranker,
    strategy_checkboxes,
    time,
):
    def run_rerank(reranker, query, docs, label):
        start = time.perf_counter()
        results = reranker.rerank(query, docs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"label": label, "latency_ms": elapsed_ms, "results": results}

    all_results = []
    if active_query and active_docs:
        if strategy_checkboxes["Hybrid Fusion"].value and hybrid_reranker:
            all_results.append(
                run_rerank(hybrid_reranker, active_query, active_docs, "Hybrid Fusion")
            )
        if strategy_checkboxes["Distilled Pairwise"].value and distilled_reranker:
            all_results.append(
                run_rerank(distilled_reranker, active_query, active_docs, "Distilled Pairwise")
            )
        if strategy_checkboxes["Shallow ColBERT"].value and colbert_reranker:
            all_results.append(
                run_rerank(colbert_reranker, active_query, active_docs, "Shallow ColBERT")
            )
    all_results
    return (all_results,)


@app.cell
def _(all_results, mo):
    def format_ranking(res_item):
        rows = []
        for r in res_item["results"]:
            doc_preview = r.doc[:100] + ("..." if len(r.doc) > 100 else "")
            rows.append(f"| {r.rank} | {r.score:.4f} | {doc_preview} |")
        header = "| Rank | Score | Document |\n|------|-------|----------|"
        return header + "\n" + "\n".join(rows)

    sections = []
    for res_item in all_results:
        sections.append(
            mo.md(
                f"## {res_item['label']}\n"
                f"**Latency:** {res_item['latency_ms']:.2f}ms\n\n"
                f"{format_ranking(res_item)}"
            )
        )
    if not sections:
        sections.append(mo.md("*No results — enter a query and documents above.*"))
    mo.vstack(sections)
    return


@app.cell
def _(all_results, mo):
    latency_rows = [f"| {r['label']} | {r['latency_ms']:.2f}ms |" for r in all_results]
    latency_table = (
        mo.md(
            "## Latency Comparison\n\n"
            "| Strategy | Latency |\n|----------|--------|\n" + "\n".join(latency_rows)
        )
        if len(all_results) >= 2
        else mo.md("")
    )
    latency_table
    return


if __name__ == "__main__":
    app.run()
