# Interactive Reranking Explorer

Compare Hybrid Fusion, Distilled Pairwise, and Shallow ColBERT reranking strategies side-by-side on custom queries and documents.

<div style="margin: 0 -0.8rem">
  <iframe src="/shallow_cross_encoders/notebooks/html/01_interactive_reranking.html"    style="width:100%; height:600px; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"    loading="lazy"></iframe>
</div>

## Run Locally

```bash
uv sync --extra docs
uv run quarto render notebooks/
```
