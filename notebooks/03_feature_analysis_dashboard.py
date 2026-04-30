# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=1.26.0",
#     "matplotlib>=3.8.0",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", app_title="Feature Analysis Dashboard")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np

    return Path, json, mo, np


@app.cell
def _(mo):
    mo.md("""
    # Feature Analysis Dashboard

    Explore the **feature vectors** that drive the Hybrid Fusion Reranker.
    Visualize semantic scores, BM25 scores, token overlap, and more —
    interactively, on your own data.
    """)
    return


@app.cell
def _():
    from reranker.embedder import Embedder
    from reranker.strategies.hybrid import HybridFusionReranker

    return Embedder, HybridFusionReranker


@app.cell
def _(Path, json):
    pairs_path = Path("data/raw/pairs.jsonl")
    pairs_data = []
    if pairs_path.exists():
        with open(pairs_path) as f:
            pairs_data = [json.loads(line) for line in f if line.strip()]
    pairs_data
    return (pairs_data,)


@app.cell
def _(Embedder, HybridFusionReranker, pairs_data):
    embedder = Embedder()
    reranker = HybridFusionReranker(embedder=embedder)
    if pairs_data:
        queries = [p["query"] for p in pairs_data]
        docs = [p["doc"] for p in pairs_data]
        scores = [float(p["score"]) for p in pairs_data]
        reranker.fit_pointwise(queries, docs, scores, use_regression=True)
    embedder, reranker
    return embedder, reranker


@app.cell
def _(mo, pairs_data):
    unique_queries = list(dict.fromkeys(p["query"] for p in pairs_data)) if pairs_data else []
    unique_docs = list(dict.fromkeys(p["doc"] for p in pairs_data)) if pairs_data else []

    query_input = mo.ui.text(
        value=unique_queries[0] if unique_queries else "python dataclass default factory",
        label="Query",
        full_width=True,
    )

    docs_textarea = mo.ui.text_area(
        value="\n".join(unique_docs[:8]) if unique_docs else "Enter documents, one per line",
        label="Documents (one per line)",
        full_width=True,
        rows=6,
    )
    query_input, docs_textarea
    return docs_textarea, query_input


@app.cell
def _(docs_textarea, query_input):
    active_query = query_input.value.strip()
    active_docs = [d.strip() for d in docs_textarea.value.strip().split("\n") if d.strip()]
    active_query, active_docs
    return active_docs, active_query


@app.cell
def _(active_docs, active_query, mo, reranker):
    X = (
        reranker._build_features(active_query, active_docs)
        if active_query and active_docs
        else None
    )
    feature_names = reranker.feature_names_

    if X is not None and len(X) > 0:
        rows = []
        for fname in feature_names:
            idx = reranker._feature_registry.get(fname)
            if idx is not None and idx < X.shape[1]:
                vals = ", ".join(f"{X[r, idx]:.4f}" for r in range(X.shape[0]))
                rows.append(f"| {fname} | {vals} |")
        header = "| Feature | " + " | ".join(f"Doc {i + 1}" for i in range(X.shape[0])) + " |\n"
        header += "|" + "--------|" * (X.shape[0] + 1)
        feature_table = mo.md("## Feature Matrix\n" + header + "\n" + "\n".join(rows))
    else:
        feature_table = mo.md("*Enter a query and documents above to see features.*")
    feature_table
    return X, feature_names


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(X, active_docs, feature_names, mo, np, plt):
    if X is not None and len(X) > 0 and X.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(max(8, X.shape[1] * 0.8), max(4, len(active_docs) * 0.6)))

        n_docs = X.shape[0]
        n_features = X.shape[1]
        bar_width = 0.8 / n_docs
        indices = np.arange(n_features)

        for doc_idx in range(n_docs):
            offset = (doc_idx - n_docs / 2 + 0.5) * bar_width
            ax.barh(indices + offset, X[doc_idx], bar_width, label=f"Doc {doc_idx + 1}")

        ax.set_yticks(indices)
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_xlabel("Feature Value")
        ax.set_title("Feature Values per Document")
        ax.legend(fontsize=7)
        plt.tight_layout()
        feature_chart = fig
    else:
        feature_chart = mo.md("*No features to plot.*")
    feature_chart
    return


@app.cell
def _(active_docs, active_query, embedder, mo, np):
    if active_query and active_docs:
        q_vec = embedder.encode([active_query])[0]
        d_vecs = embedder.encode(active_docs)
        sem_scores = [float(np.dot(q_vec, d)) for d in d_vecs]
        max_score = max(sem_scores) if sem_scores else 1.0
        top_idx = sem_scores.index(max_score)

        sem_output = mo.md(
            "## Semantic Similarity\n\n"
            f"**Query:** `{active_query}`\n\n"
            + "\n".join(
                f"- Doc {i + 1}: **{s:.4f}** {'<-- best' if i == top_idx else ''}\n"
                f"  > {active_docs[i][:120]}"
                for i, s in enumerate(sem_scores)
            )
        )
    else:
        sem_output = mo.md("")
    sem_output
    return


@app.cell
def _(active_docs, active_query, embedder, plt):
    if active_query and active_docs and len(active_docs) >= 2:
        all_texts = [active_query] + active_docs
        vecs = embedder.encode(all_texts)

        from scipy.spatial.distance import cdist

        dist_matrix = cdist(vecs, vecs, metric="cosine")

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        labels = ["QUERY"] + [f"Doc {i + 1}" for i in range(len(active_docs))]
        im = ax2.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")
        ax2.set_xticks(range(len(labels)))
        ax2.set_yticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.set_title("Pairwise Cosine Distance Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax2.text(j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig2.colorbar(im, ax=ax2, shrink=0.8)
        plt.tight_layout()
        dist_chart = fig2
    else:
        dist_chart = None
    dist_chart
    return


@app.cell
def _(mo, pairs_data):
    if pairs_data:
        score_dist = {}
        for p in pairs_data:
            s = p["score"]
            score_dist[s] = score_dist.get(s, 0) + 1
        dist_rows = "\n".join(
            f"| {score} | {count} | {'█' * count}" for score, count in sorted(score_dist.items())
        )
        label_output = mo.md(
            "## Dataset Label Distribution\n\n"
            "| Score | Count | Bar |\n|-------|-------|-----|\n" + dist_rows
        )
    else:
        label_output = mo.md("")
    label_output
    return


if __name__ == "__main__":
    app.run()
