---
hide:
  - navigation
  - toc
---

# Architecture & Live Reranking

Walk through the protocol layer, embedder internals, then compare all reranking strategies side-by-side on real data.

<div class="iframe-container" id="iframe-wrapper-interactive-reranking">
  <div class="iframe-controls">
    <button onclick="toggleNotebookFullscreen(this)" class="md-button">Expand</button>
    <a href="/shallow_cross_encoders/notebooks/html/01_interactive_reranking.html" target="_blank" class="md-button">Open in New Tab</a>
  </div>
  <iframe src="/shallow_cross_encoders/notebooks/html/01_interactive_reranking.html" allowfullscreen loading="lazy"></iframe>
</div>

## Run Locally

```bash
uv sync --extra docs
uv run quarto render notebooks/
```
