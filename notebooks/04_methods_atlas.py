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
app = marimo.App(width="full", app_title="Methods Overview & Benchmark Atlas")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    return mo, patheffects, plt


@app.cell
def _(mo):
    mo.md("""
    # 🗺️ Reranking Methods Atlas

    A self-contained reference for every reranking strategy in this library —
    with methodology, benchmark stats, pros/cons, and interactive comparisons.

    All benchmarks on: **potion-base-32M** · synthetic test set · seed 42 · n=20 docs
    """)
    return


@app.cell
def _():
    B = {
        "bm25": {
            "name": "BM25",
            "cls": "BM25Engine",
            "file": "reranker/lexical.py",
            "type": "Lexical",
            "ndcg": 0.0861,
            "map": 0.0500,
            "mrr": 0.0500,
            "p1": 0.0,
            "latency_ms": 0.08,
            "p50_ms": 0.07,
            "p99_ms": 0.17,
            "qps": 11854,
            "cold_start_ms": 0.0,
            "bm25_uplift": 0.0,
            "pros": [
                "Fastest non-binary strategy (11854 QPS)",
                "No embedding model needed",
                "Zero cold-start time",
                "Deterministic and interpretable",
            ],
            "cons": [
                "Lexical only — 'car' ≠ 'automobile'",
                "No semantic understanding",
                "Lowest NDCG@10 (0.0861)",
                "No phrase or field weighting",
            ],
            "best_for": "Baseline, exact-term matching, resource-constrained",
            "summary": "The classic bag-of-words ranking. Fast and reliable for exact term matches, but misses all semantic relationships.",
        },
        "hybrid": {
            "name": "Hybrid Fusion",
            "cls": "HybridFusionReranker",
            "file": "reranker/strategies/hybrid.py",
            "type": "GBDT",
            "ndcg": 0.2000,
            "map": 0.2000,
            "mrr": 0.2000,
            "p1": 0.2000,
            "latency_ms": 1.07,
            "p50_ms": 1.01,
            "p99_ms": 1.38,
            "qps": 932,
            "cold_start_ms": 119.3,
            "bm25_uplift": 0.0738,
            "formula": "S = (S_model + S_blend) / 2",
            "features": "9 base: sem, bm25, overlap, coverage, phrase, lens, norm_diff",
            "pros": [
                "Best quality on synthetic data (NDCG=0.2000)",
                "Interpretable heuristic blend + learned model",
                "Extensible via HeuristicAdapter protocol",
                "Best scaling: 3.7x latency for 10x corpus",
            ],
            "cons": [
                "Requires training data",
                "Cold-start ~119ms for model load",
                "Moderate throughput (932 QPS)",
                "Adapter features add latency per doc",
            ],
            "best_for": "Production accuracy-critical, when training data available",
            "summary": "A GBDT that fuses semantic, lexical, and heuristic features. Combines XGBoost with a weighted heuristic blend. The most accurate strategy on synthetic benchmarks.",
        },
        "binary": {
            "name": "Binary Quantized",
            "cls": "BinaryQuantizedReranker",
            "file": "reranker/strategies/binary_reranker.py",
            "type": "Binary",
            "ndcg": 0.2000,
            "map": 0.2000,
            "mrr": 0.2000,
            "p1": 0.2000,
            "latency_ms": 0.04,
            "p50_ms": 0.04,
            "p99_ms": 0.07,
            "qps": 22866,
            "cold_start_ms": 0.1,
            "bm25_uplift": 0.0738,
            "formula": "S = popcount(q_bin ⊗ d_bin) (Hamming)",
            "pros": [
                "Fastest strategy: 22,866 QPS at 0.04ms",
                "Near-zero cold-start (0.1ms)",
                "Best accuracy/latency trade-off overall",
                "Memory efficient: binary vectors = 1/8 of float32",
            ],
            "cons": [
                "Bilinear refinement adds no gain on synthetic data",
                "Accuracy ceiling at high NDCG levels",
                "Requires embedding model",
                "Binary thresholding loses fine-grained signal",
            ],
            "best_for": "Latency-critical, high-throughput, real-time production",
            "summary": "Quantizes embeddings to binary vectors and scores via Hamming distance. The fastest strategy by a wide margin while matching hybrid quality on synthetic data.",
        },
        "distilled": {
            "name": "Distilled Pairwise",
            "cls": "DistilledPairwiseRanker",
            "file": "reranker/strategies/distilled.py",
            "type": "Logistic",
            "ndcg": 0.2000,
            "map": 0.2000,
            "mrr": 0.2000,
            "p1": 0.2000,
            "latency_ms": 0.10,
            "p50_ms": 0.10,
            "p99_ms": 0.14,
            "qps": None,
            "cold_start_ms": None,
            "bm25_uplift": None,
            "accuracy": 1.0,
            "formula": "P(A≻B|q) = σ(w·x + b)",
            "features": "7 pairwise: sim(a), sim(b), sim_diff, dist, len_a, len_b, len_diff",
            "pros": [
                "100% pairwise accuracy on synthetic data",
                "Ultra-fast (0.10ms per comparison)",
                "Captures LLM judgment at zero API cost",
                "Merge-sort algorithm for large sets (O(n log n))",
            ],
            "cons": [
                "Produces preferences, not absolute scores",
                "Tournament O(n²) for small sets",
                "Transitivity not guaranteed — possible cycles",
                "High accuracy variance (±34.7%) on harder data",
            ],
            "best_for": "Pairwise comparison, LLM distillation at inference",
            "summary": "Logistic regression trained on LLM preference judgments. Conducts pairwise tournaments to produce rankings — perfect on synthetic, but watch for variance on real data.",
        },
        "colbert": {
            "name": "Shallow ColBERT",
            "cls": "StaticColBERTReranker",
            "file": "reranker/strategies/late_interaction.py",
            "type": "Late Interaction",
            "ndcg": 0.1262,
            "map": 0.1000,
            "mrr": 0.1000,
            "p1": 0.0,
            "latency_ms": 0.37,
            "p50_ms": 0.37,
            "p99_ms": 0.43,
            "qps": 2710,
            "cold_start_ms": 0.3,
            "bm25_uplift": 0.0,
            "formula": "Score = Σqₜ maxdₜ cos(qₜ, dₜ)",
            "pros": [
                "Token-level MaxSim captures fine-grained alignment",
                "Fast: 2,710 QPS at 0.37ms p50",
                "Salience-based token pruning",
                "Quantization support (4-bit / ternary)",
            ],
            "cons": [
                "Lower quality than hybrid/binary on synthetic data",
                "No BM25 uplift on current benchmarks",
                "P@1 = 0 on synthetic test set",
                "Requires pre-built token index",
            ],
            "best_for": "When token-level match matters, as ensemble component",
            "summary": "Token-level late interaction via MaxSim. Stores per-token embeddings for fine-grained query-doc matching. Fast but lower accuracy on synthetic benchmarks.",
        },
        "consistency": {
            "name": "Consistency Engine",
            "cls": "ConsistencyEngine",
            "file": "reranker/strategies/consistency.py",
            "type": "Structured",
            "latency_ms": 0.29,
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "fpr": 0.0,
            "formula": "Check: entity+attr ≈ entity'+attr', then val conflict?",
            "pros": [
                "Perfect recall, precision, F1 on synthetic data",
                "Zero false positives at all thresholds",
                "Fast: 0.29ms per check",
                "Structured + fuzzy claim detection",
            ],
            "cons": [
                "Only detects factual contradictions",
                "Requires extractable structured claims",
                "Strictly a consistency tool, not general ranking",
                "Regex-based extraction misses implicit claims",
            ],
            "best_for": "Fact-checking, document consistency verification",
            "summary": "Detects contradictions across documents using structured claim extraction + semantic alignment. Perfect benchmark scores make it a reliable drop-in for fact-verification pipelines.",
        },
        "cascade": {
            "name": "Cascade Reranker",
            "cls": "CascadeReranker",
            "file": "reranker/strategies/cascade.py",
            "type": "Cascade",
            "ndcg": 0.2000,
            "map": 0.2000,
            "mrr": 0.2000,
            "p1": 0.2000,
            "latency_ms": 0.94,
            "p50_ms": 0.93,
            "p99_ms": 1.00,
            "qps": 1060,
            "cold_start_ms": 0.0,
            "bm25_uplift": 0.0738,
            "pros": [
                "Matches hybrid quality with safety net",
                "Slightly faster than hybrid alone (0.93 vs 1.01ms)",
                "Fallback to FlashRank on low confidence",
                "Configurable confidence threshold",
            ],
            "cons": [
                "0% fallback rate on synthetic — all handled by primary",
                "FlashRank fallback is 20x slower when triggered",
                "Requires FlashRank dependency",
                "Adds complexity over standalone hybrid",
            ],
            "best_for": "Production when you want hybrid + safety net",
            "summary": "A two-stage cascade: primary (hybrid) + fallback (FlashRank). Activates fallback only when hybrid confidence is below threshold.",
        },
    }
    SCALING = [
        (20, 0.07, 1.10, 0.33, 0.05),
        (50, 0.07, 0.92, 0.30, 0.05),
        (100, 0.06, 0.86, 0.28, 0.04),
        (200, 0.06, 0.86, 0.25, 0.04),
    ]
    EMBEDDER_GRID = [
        ("potion-base-8M", 64, "binary_reranker", 1.0000, 0.02),
        ("potion-base-8M", 128, "binary_reranker", 1.0000, 0.02),
        ("potion-base-8M", 256, "binary_reranker", 1.0000, 0.02),
        ("potion-base-8M", 512, "binary_reranker", 1.0000, 0.02),
        ("potion-base-32M", 64, "hybrid", 1.0000, 0.67),
        ("potion-base-32M", 64, "binary_reranker", 1.0000, 0.02),
        ("potion-base-32M", 128, "hybrid", 1.0000, 0.68),
        ("potion-base-32M", 128, "binary_reranker", 1.0000, 0.02),
        ("potion-base-32M", 256, "hybrid", 1.0000, 0.63),
        ("potion-base-32M", 256, "binary_reranker", 1.0000, 0.02),
        ("potion-base-32M", 512, "hybrid", 1.0000, 0.65),
        ("potion-base-32M", 512, "binary_reranker", 1.0000, 0.02),
        ("potion-base-8M", 64, "hybrid", 0.9262, 0.66),
        ("potion-base-8M", 64, "late_interaction", 0.9262, 0.19),
        ("potion-base-8M", 128, "hybrid", 0.9262, 0.66),
        ("potion-base-8M", 128, "late_interaction", 0.9262, 0.17),
        ("potion-base-8M", 256, "hybrid", 0.9262, 0.64),
        ("potion-base-8M", 256, "late_interaction", 0.9262, 0.17),
        ("potion-base-8M", 512, "hybrid", 0.9262, 0.67),
        ("potion-base-8M", 512, "late_interaction", 0.9262, 0.16),
        ("potion-base-32M", 64, "late_interaction", 0.9262, 0.19),
        ("potion-base-32M", 128, "late_interaction", 0.9262, 0.17),
        ("potion-base-32M", 256, "late_interaction", 0.9262, 0.17),
        ("potion-base-32M", 512, "late_interaction", 0.9262, 0.18),
        ("potion-multilingual-128M", 64, "hybrid", 0.9262, 0.68),
        ("potion-multilingual-128M", 64, "binary_reranker", 0.9262, 0.02),
        ("potion-multilingual-128M", 64, "late_interaction", 0.9262, 0.18),
        ("potion-multilingual-128M", 128, "hybrid", 0.9262, 0.79),
        ("potion-multilingual-128M", 128, "binary_reranker", 0.9262, 0.02),
        ("potion-multilingual-128M", 128, "late_interaction", 0.9262, 0.18),
        ("potion-multilingual-128M", 256, "hybrid", 0.9262, 0.73),
        ("potion-multilingual-128M", 256, "binary_reranker", 0.9262, 0.02),
        ("potion-multilingual-128M", 256, "late_interaction", 0.9262, 0.18),
        ("potion-multilingual-128M", 512, "hybrid", 0.9262, 0.77),
        ("potion-multilingual-128M", 512, "binary_reranker", 0.9262, 0.02),
        ("potion-multilingual-128M", 512, "late_interaction", 0.9262, 0.17),
    ]
    return B, EMBEDDER_GRID, SCALING


@app.cell
def _(B, mo):
    _ranking_keys = ["hybrid", "binary", "distilled", "cascade", "colbert", "bm25"]
    _speed_keys = ["binary", "bm25", "colbert", "cascade", "hybrid", "distilled"]
    _fmt = lambda v, f: f"{v:{f}}" if v is not None else "—"
    _ranking_rows = [
        f"| {B[k]['name']} | {B[k]['type']} | {B[k]['ndcg']:.4f} | {B[k]['mrr']:.4f} | {_fmt(B[k]['qps'], ',')} |"
        for k in _ranking_keys
    ]
    _ranking_header = (
        "| Strategy | Type | NDCG@10 | MRR | QPS |\n|----------|------|---------|-----|-----|"
    )
    _speed_rows = [
        f"| {B[k]['name']} | {B[k]['latency_ms']:.2f} | {_fmt(B[k]['p50_ms'], '.2f')} | {_fmt(B[k]['qps'], ',')} |"
        for k in _speed_keys
    ]
    _speed_header = (
        "| Strategy | Latency (ms) | p50 | QPS |\n|----------|-------------|-----|-----|"
    )
    _c = B["consistency"]
    _cline = f"Recall={_c['recall']}, Precision={_c['precision']}, F1={_c['f1']}, FPR={_c['fpr']}, Latency={_c['latency_ms']}ms"

    mo.vstack(
        [
            mo.md("## Strategy Quick Reference"),
            mo.md("### Ranking Quality"),
            mo.md(_ranking_header + "\n" + "\n".join(_ranking_rows)),
            mo.md("### Speed"),
            mo.md(_speed_header + "\n" + "\n".join(_speed_rows)),
            mo.md(f"### Consistency Engine\n\n{_cline}"),
        ]
    )
    return


@app.cell
def _(B, mo):
    strategy_names = sorted(B.keys())
    strat_dropdown = mo.ui.dropdown(
        options={f"{B[k]['name']} ({k})": k for k in strategy_names},
        value="Hybrid Fusion (hybrid)",
        label="Pick a strategy for details",
    )
    strat_dropdown
    return (strat_dropdown,)


@app.cell
def _(B, mo, strat_dropdown):
    cur_key = strat_dropdown.value
    cur_s = B[cur_key]
    md_parts = [f"# {cur_s['name']} (`{cur_s['cls']}`)\n"]
    md_parts.append(f"**File:** `{cur_s['file']}` · **Type:** {cur_s['type']}\n")
    md_parts.append(f"_{cur_s['summary']}_\n")
    if cur_s.get("formula"):
        md_parts.append(f"\n### Core Formula\n\n`{cur_s['formula']}`\n")
    if cur_s.get("features"):
        md_parts.append(f"\n### Features\n\n{cur_s['features']}\n")
    md_parts.append("\n### Pros\n")
    for p_item in cur_s["pros"]:
        md_parts.append(f"- ✅ {p_item}")
    md_parts.append("\n### Cons\n")
    for c_item in cur_s["cons"]:
        md_parts.append(f"- ⚠️ {c_item}")
    md_parts.append(f"\n**Best for:** _{cur_s['best_for']}_\n")
    score_keys = [
        "ndcg",
        "map",
        "mrr",
        "p1",
        "latency_ms",
        "p50_ms",
        "qps",
        "bm25_uplift",
        "recall",
        "precision",
        "f1",
        "accuracy",
    ]
    stats = {sk: sv for sk, sv in cur_s.items() if sk in score_keys and sv is not None}
    if stats:
        md_parts.append("\n### Benchmark Stats\n")
        for sk_name, sv_val in stats.items():
            sk_label = sk_name.replace("_", " ").title()
            if isinstance(sv_val, float):
                md_parts.append(
                    f"- **{sk_label}:** `{sv_val:.4f}`"
                    if sv_val < 10
                    else f"- **{sk_label}:** `{sv_val:,.0f}`"
                )
            else:
                md_parts.append(f"- **{sk_label}:** `{sv_val}`")
    mo.md("\n".join(md_parts))
    return


@app.cell
def _(B, patheffects, plt):
    _keys = ["bm25", "colbert", "distilled", "cascade", "hybrid", "binary"]
    _pdata = [(B[k]["name"], B[k]["ndcg"], B[k]["latency_ms"]) for k in _keys]
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _colors = ["#888888", "#4A90D9", "#50C878", "#FFB347", "#E8575A", "#9B59B6"]
    for (_name, _ndcg, _lat), _c in zip(_pdata, _colors):
        _size = 150 + (_ndcg * 300) if _ndcg else 80
        _ax.scatter(
            _lat, _ndcg, c=_c, s=_size, label=_name, edgecolors="white", linewidth=1.5, zorder=5
        )
        _ax.annotate(
            _name,
            (_lat, _ndcg),
            textcoords="offset points",
            xytext=(8, 6 if _name != "BM25" else -14),
            fontsize=9,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
        )
    _ax.set_xlabel("Latency (ms) ← faster")
    _ax.set_ylabel("NDCG@10 → better ↑")
    _ax.set_title("Pareto Frontier: Quality vs Speed")
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(-0.2, max(d[2] for d in _pdata) * 1.25)
    _ax.set_ylim(0, 0.28)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(SCALING, plt):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _sizes = [s[0] for s in SCALING]
    _ax.plot(_sizes, [s[1] for s in SCALING], "o-", label="BM25")
    _ax.plot(_sizes, [s[2] for s in SCALING], "s-", label="Hybrid")
    _ax.plot(_sizes, [s[3] for s in SCALING], "^-", label="ColBERT")
    _ax.plot(_sizes, [s[4] for s in SCALING], "D-", label="Binary")
    _ax.set_xlabel("Corpus Size (documents)")
    _ax.set_ylabel("Latency (ms)")
    _ax.set_title("Latency Scaling with Corpus Size")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(EMBEDDER_GRID, mo):
    embedder_models = list(dict.fromkeys(e[0] for e in EMBEDDER_GRID))
    embedder_strategies = list(dict.fromkeys(e[2] for e in EMBEDDER_GRID))
    model_dd = mo.ui.dropdown(options=embedder_models, value=embedder_models[0], label="Model")
    strategy_dd = mo.ui.dropdown(
        options=embedder_strategies, value=embedder_strategies[0], label="Strategy"
    )
    mo.vstack(
        [
            mo.md("## Embedder Model Comparison"),
            mo.vstack([model_dd, strategy_dd]),
        ]
    )
    return model_dd, strategy_dd


@app.cell
def _(EMBEDDER_GRID, mo, model_dd, strategy_dd):
    filtered = [e for e in EMBEDDER_GRID if e[0] == model_dd.value and e[2] == strategy_dd.value]
    embedder_rows = [f"| {e[1]} | {e[3]:.4f} | {e[4]:.2f} |" for e in filtered]
    embedder_table = (
        mo.md(
            f"### {model_dd.value} · {strategy_dd.value}\n\n"
            + "| Dim | NDCG@10 | Latency (ms) |\n|-----|---------|-------------|\n"
            + "\n".join(embedder_rows)
        )
        if embedder_rows
        else mo.md("*No results for this combination.*")
    )
    embedder_table
    return


@app.cell
def _(B, plt):
    _keys_with_ndcg = [k for k in B if B[k].get("ndcg") is not None]
    _names = [B[k]["name"] for k in _keys_with_ndcg]
    _ndcgs = [B[k]["ndcg"] for k in _keys_with_ndcg]
    _colors = ["#9B59B6", "#E8575A", "#50C878", "#FFB347", "#4A90D9", "#888888"]
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _bars = _ax.barh(_names, _ndcgs, color=_colors, edgecolor="white", height=0.6)
    for _bar, _val in zip(_bars, _ndcgs):
        _ax.text(
            _bar.get_width() + 0.005,
            _bar.get_y() + _bar.get_height() / 2,
            f"{_val:.4f}",
            va="center",
            fontsize=9,
        )
    _ax.set_xlabel("NDCG@10")
    _ax.set_title("Ranking Quality by Strategy")
    _ax.set_xlim(0, max(_ndcgs) * 1.3)
    _ax.grid(True, alpha=0.3, axis="x")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(B, mo):
    mo.md(
        "## Key Findings\n"
        + f"- **Best quality:** {B['hybrid']['name']} — NDCG@10=0.2000, MRR=0.2000, P@1=0.2000"
        + f"\n- **Fastest:** {B['binary']['name']} — 22,866 QPS at 0.04ms p50"
        + f"\n- **Best trade-off:** {B['binary']['name']} — matches hybrid quality at 22x lower latency"
        + f"\n- **Consistency:** {B['consistency']['name']} — perfect recall/precision/F1 on synthetic"
        + f"\n- **Pairwise:** {B['distilled']['name']} — 100% pairwise accuracy at 0.10ms per comparison"
        + "\n- **Scaling:** All strategies scale sub-linearly; BM25 & Binary are near-constant regardless of corpus size"
        + "\n- **Embedder:** potion-base-8M at dim=64 achieves same NDCG as 32M at dim=512"
    )
    return


if __name__ == "__main__":
    app.run()
