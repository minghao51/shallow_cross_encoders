"""Visualization utilities for benchmark results.

Generates Pareto frontier plots, radar charts, and comparison tables.
All plotting is guarded — missing matplotlib produces a warning instead of error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reranker.eval.statistics import bootstrap_ci


def _check_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def _get_colors(n: int) -> list[str]:
    base = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    if n <= len(base):
        return base[:n]
    return [f"#{i * 37 % 256:02x}{i * 91 % 256:02x}{i * 53 % 256:02x}" for i in range(n)]


def plot_pareto_frontier(
    results: list[dict[str, Any]],
    output_path: str | Path,
    latency_key: str = "latency_mean_ms",
    accuracy_key: str = "ndcg@10",
) -> str | None:
    """Generate latency--accuracy Pareto frontier scatter plot.

    Each point is one experiment result (strategy × configuration). The
    Pareto frontier line connects non-dominated points.

    Args:
        results: List of experiment result dicts with latency and accuracy keys.
        output_path: Where to save the PNG.
        latency_key: Key for latency in ms (x-axis).
        accuracy_key: Key for NDCG or other accuracy metric (y-axis).

    Returns:
        Path to generated PNG, or ``None`` if matplotlib is unavailable.
    """
    if not _check_matplotlib():
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points: list[tuple[float, float, str]] = []
    for r in results:
        lat = r.get("metrics", {}).get(latency_key, 0)
        acc = r.get("metrics", {}).get(accuracy_key, 0)
        name = r.get("experiment_name", r.get("strategy", "unknown"))
        if lat > 0:
            points.append((lat, acc, name))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for lat, acc, name in points:
        ax.scatter(lat, acc, s=60, alpha=0.7)
        ax.annotate(name.split("_")[0], (lat, acc), fontsize=7, alpha=0.8)

    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[float, float]] = []
    max_acc = -1.0
    for lat, acc, _ in sorted_pts:
        if acc > max_acc:
            frontier.append((lat, acc))
            max_acc = acc

    if frontier:
        fx, fy = zip(*frontier, strict=False)
        ax.plot(fx, fy, "r--", linewidth=1.5, alpha=0.6, label="Pareto frontier")

    ax.set_xlabel(f"{latency_key.replace('_', ' ').title()} (ms)")
    ax.set_ylabel(f"{accuracy_key.replace('_', ' ').title()}")
    ax.set_title("Latency–Accuracy Pareto Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def plot_radar(
    results: list[dict[str, Any]],
    output_path: str | Path,
    metrics: list[str] | None = None,
) -> str | None:
    """Generate a radar chart comparing strategies across multiple metrics.

    Args:
        results: List of experiment result dicts.
        output_path: Where to save the PNG.
        metrics: Metric keys to include. Defaults to NDCG, MAP, MRR, P@1, (inverse) latency.

    Returns:
        Path to generated PNG, or ``None`` if matplotlib is unavailable.
    """
    if not _check_matplotlib():
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics is None:
        metrics = ["ndcg@10", "map@10", "mrr", "p@1", "latency_mean_ms"]

    baseline_results = [
        r
        for r in results
        if "baseline" in r.get("experiment_name", "") and "ndcg@10" in r.get("metrics", {})
    ]
    if not baseline_results:
        return None

    strategies = [r.get("strategy", "unknown") for r in baseline_results]
    unique_strategies = list(dict.fromkeys(strategies))
    n_strats = len(unique_strategies)
    if n_strats < 2:
        return None

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    colors = _get_colors(n_strats)
    for idx, strat in enumerate(unique_strategies):
        strat_results = [r for r in baseline_results if r.get("strategy") == strat]
        if not strat_results:
            continue
        values: list[float] = []
        for m in metrics:
            vals = [r.get("metrics", {}).get(m, 0) for r in strat_results]
            avg = sum(vals) / len(vals) if vals else 0.0
            if m == "latency_mean_ms":
                avg = 1.0 / max(avg, 0.001)
            values.append(avg)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=strat, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    metric_labels = [m.replace("_", " ").title() for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title("Strategy Comparison Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def generate_comparison_table(
    results: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> str:
    """Generate a rich markdown comparison table with confidence intervals.

    Args:
        results: List of experiment result dicts from ``save_results``.
        output_path: Optional path to write the markdown table.

    Returns:
        Markdown string of the comparison table.
    """
    baseline_results = [
        r
        for r in results
        if "baseline" in r.get("experiment_name", "") and "ndcg@10" in r.get("metrics", {})
    ]

    if not baseline_results:
        return ""

    strat_metrics: dict[str, dict[str, list[float]]] = {}
    for r in baseline_results:
        strat = r.get("strategy", "unknown")
        if strat not in strat_metrics:
            strat_metrics[strat] = {}
        metrics = r.get("metrics", {})
        for key in ["ndcg@10", "map@10", "mrr", "p@1", "latency_mean_ms", "throughput_qps"]:
            if key in metrics:
                strat_metrics[strat].setdefault(key, []).append(metrics[key])

    lines = []
    lines.append("## Strategy Comparison with 95% Confidence Intervals")
    lines.append("")
    lines.append(
        "| Strategy | NDCG@10 | NDCG@10 CI 95% | MAP@10 | MRR | P@1 | Lat (ms) | Throughput |"  # noqa: E501
    )
    lines.append(
        "|----------|---------|----------------|--------|-----|-----|----------|------------|"
    )

    for strat, metrics_dict in strat_metrics.items():
        ndcg_vals = metrics_dict.get("ndcg@10", [])
        ndcg_mean = sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else 0.0
        ndcg_ci = "—"
        if len(ndcg_vals) >= 2:
            lo, hi = bootstrap_ci(ndcg_vals)
            ndcg_ci = f"[{lo:.4f}, {hi:.4f}]"

        map_val = (
            sum(metrics_dict.get("map@10", [0])) / len(metrics_dict.get("map@10", [0]))
            if metrics_dict.get("map@10")
            else 0.0
        )
        mrr_val = (
            sum(metrics_dict.get("mrr", [0])) / len(metrics_dict.get("mrr", [0]))
            if metrics_dict.get("mrr")
            else 0.0
        )
        p1_val = (
            sum(metrics_dict.get("p@1", [0])) / len(metrics_dict.get("p@1", [0]))
            if metrics_dict.get("p@1")
            else 0.0
        )
        lat_val = (
            sum(metrics_dict.get("latency_mean_ms", [0]))
            / len(metrics_dict.get("latency_mean_ms", [0]))
            if metrics_dict.get("latency_mean_ms")
            else 0.0
        )
        qps_val = (
            sum(metrics_dict.get("throughput_qps", [0]))
            / len(metrics_dict.get("throughput_qps", [0]))
            if metrics_dict.get("throughput_qps")
            else 0.0
        )

        lines.append(
            f"| {strat} | {ndcg_mean:.4f} | {ndcg_ci} | {map_val:.4f} | {mrr_val:.4f} | {p1_val:.4f} | {lat_val:.2f} | {qps_val:.0f} |"  # noqa: E501
        )

    lines.append("")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    return "\n".join(lines)
