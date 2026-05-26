# Project State

Last updated: 2026-05-12

## Implemented Features

| Module / Strategy | Status | Key Files |
|---|---|---|
| **BinaryQuantizedReranker** (Hamming + bilinear) | Complete — fit, score, rerank, save/load | `src/reranker/strategies/binary_reranker.py` |
| **CascadeReranker** (confidence-based fallback) | Complete — 4 confidence metrics, 3 fallback strategies, stats tracking | `src/reranker/strategies/cascade.py` |
| **ConsistencyEngine** (contradiction detection) | Complete — 19 structured claim patterns, fuzzy attribute matching, save/load, diagnose_misses | `src/reranker/strategies/consistency.py` |
| **DistilledPairwiseRanker** (LR / Cross-Encoder) | Complete — pairwise, listwise (ListMLE), LambdaLoss; full tournament + merge-sort for large doc sets | `src/reranker/strategies/distilled.py` |
| **FlashRankEnsemble / FlashRankWrapper** | Complete — multi-model ensemble averaging | `src/reranker/strategies/flashrank_ensemble.py` |
| **HardNegativeFlashRankEnsemble** | Complete — extends FlashRankEnsemble with hard-negative scoring | `src/reranker/strategies/flashrank_ensemble.py:157` |
| **HybridFusionReranker** (XGBoost/sklearn) | Complete — fit, fit_pointwise, rerank_batch, save/load (xgboost JSON + legacy pickle), MetaRouter integration | `src/reranker/strategies/hybrid.py` |
| **MetaRouter** (query-type routing) | Complete — DecisionTree/MLP, 3 weight profiles (navigational/informational/balanced), predict + predict_proba | `src/reranker/strategies/meta_router.py` |
| **MultiReranker** (RRF fusion) | Complete — weights, arbitrary number of rerankers; save/load intentionally stubbed (wraps instances) | `src/reranker/strategies/multi.py` |
| **PipelineReranker** (multi-stage cascade) | Complete — arbitrary stage chaining, latency tracking, save/load | `src/reranker/strategies/pipeline.py` |
| **SPLADEReranker** (sparse encoder) | Complete — fit, score, rerank, save/load | `src/reranker/strategies/splade.py` |
| **StaticColBERTReranker** (late interaction) | Complete — token-level MaxSim, salience TF-IDF pruning, quantization (4bit/ternary/int8/float16), rerank_batch, save/load | `src/reranker/strategies/late_interaction.py` |
| **SentenceTransformerWrapper** (cross-encoder) | Complete | `src/reranker/strategies/flashrank_ensemble.py:116` |
| **Embedder** (model2vec + hashed fallback) | Complete — automatic fallback, caching, CJK-aware tokenization | `src/reranker/embedder.py` |
| **EmbeddingCache** (TTLCache) | Complete — thread-safe, shared global instance, invalidation | `src/reranker/embedding_cache.py` |
| **BM25Engine** (rank_bm25 + pure Python) | Complete — incremental update, remove, fit, score, normalize; fallback BM25Okapi | `src/reranker/lexical.py` |
| **Heuristic Adapters** (KeywordMatchAdapter, LSHAdapter) | Complete | `src/reranker/heuristics/` |
| **Quantization** (4-bit, ternary, int8, float16) | Complete — quantize/dequantize, compression_ratio, memory_bytes | `src/reranker/quantization.py` |
| **Persistence** (safe joblib+JSON, legacy pickle) | Complete — artifact metadata validation, format versioning, security-warned fallback | `src/reranker/persistence.py` |
| **Settings** (Pydantic frozen, env vars, YAML) | Complete — 19 sub-configs, contextvar overrides, deep-merge from YAML | `src/reranker/config.py` |
| **Synthetic Data Generation** (pairs, preferences, contradictions, expanded data, enhanced generator) | Complete — LLM-based generation with LiteLLM/OpenRouter | `src/reranker/data/synth/` |
| **Evaluation Framework** (metrics, profiling, statistics, visualization, benchmark runner) | Complete — NDCG, MRR, Recall, latency profiling, ROI estimation | `src/reranker/eval/` |
| **CLI** (benchmark, eval, train, serve, doctor, generate) | Complete — typer-based | `src/reranker/cli/` |

## Stubbed / Unimplemented

| Item | File:Line | Severity | Notes |
|---|---|---|---|
| `MultiReranker.save()` / `.load()` — `NotImplementedError` | `src/reranker/strategies/multi.py:107,114` | Low | Intentional by design — wraps instances. Docs say "Save individual rerankers instead." |
| `except Exception: pass` — LiteLLM stream error | `src/reranker/data/client.py:201,207` | Medium | Silent swallow on streaming errors. At minimum should log. |
| `except Exception: pass` — Tokenizer fallback | `src/reranker/embedder.py:170` | Low | Intentional — attempts tokenizer, falls through to simple_tokenize. |
| **Active Distillation** — disabled by default | `src/reranker/config.py:194` | Low | `enabled: bool = False`. Code exists but inactive until explicitly enabled. |
| **MetaRouter** — disabled by default | `src/reranker/config.py:171` | Low | `enabled: bool = False`. Code exists but inactive until explicitly enabled. |
| **LSH** — disabled by default | `src/reranker/config.py:183` | Low | `enabled: bool = False`. Code exists but inactive until explicitly enabled. |

## Known Bugs

No `BUG`, `FIXME`, `TODO`, `XXX`, `HACK`, or `WORKAROUND` markers found anywhere in the source code (`src/`, `tests/`, `scripts/`, `benchmarks/`). The codebase is exceptionally clean in this regard — no open known bugs are tracked inline.

## Security Concerns

| Issue | File:Line | Severity | Notes |
|---|---|---|---|
| **Broad `except Exception` handlers (19 instances)** | See below | High | Silent error swallowing across the codebase |
| `except Exception` — optional import fallbacks | `src/reranker/utils.py:16`, `src/reranker/embedding_cache.py:12` | Low | Acceptable for optional dependency guards |
| `except Exception` — model loading fallback | `src/reranker/embedder.py:75,101,169` | Medium | Swallows model2vec init/tokenizer failures silently |
| `except Exception` — training failure | `src/reranker/strategies/binary_reranker.py:92`, `src/reranker/strategies/distilled.py:365`, `src/reranker/strategies/flashrank_ensemble.py:169` | Medium | Falls back to uniform weights; logs warning |
| `except Exception` — synthetic data generation | `src/reranker/data/synth/generator/enhanced.py:121,226,308`, `preferences.py:91`, `pairs.py:110`, `contradictions.py:123` | Medium | Silently skips failed LLM generations |
| `except Exception` — profiling fallback | `src/reranker/eval/profiling.py:38,79` | Low | Acceptable for optional profiling dependencies |
| `except Exception` — custom BEIR loading | `src/reranker/data/custom_beir.py:50` | Medium | Broad catch during data loading |
| `except Exception` — CLI training | `src/reranker/cli/train.py:320` | Medium | Broad catch wraps training loop |
| `except Exception` — deps check | `src/reranker/deps.py:37` | Low | Acceptable for import checks |
| **API keys in config objects** | `src/reranker/config.py:56,202` | Low | Properly read from env vars (`OPENROUTER_API_KEY`, `LITELLM_API_KEY`), not hardcoded. No hardcoded secrets found. |
| **Legacy pickle loading** | `src/reranker/persistence.py:97-117` | Low | Gated behind `RERANKER_ALLOW_LEGACY_PICKLE` env var; emits security warning. |

## Performance Issues

| Issue | File:Line | Severity | Notes |
|---|---|---|---|
| **Full-document embedding at score time** | `src/reranker/strategies/binary_reranker.py:156` | Low | Re-encodes query on every `score()` call; fine for single-query. |
| **O(n²) pairwise comparison for full tournament** | `src/reranker/strategies/distilled.py:289-313` | Medium | Full O(n²) for docs up to `full_tournament_max_docs` (default 50). Gated by config but could spike latency. Merge-sort path (line 315) is O(n log n) but still calls `compare()` per merge step. |
| **Synchronous batch encoding** | `src/reranker/embedder.py:137-156` | Low | `encode()` processes entire list synchronously. No async or streaming support. |
| **EmbeddingCache lock contention** | `src/reranker/embedding_cache.py:51,68` | Low | Thread-safe but fine-grained locking. Unlikely bottleneck. |
| **LSH n-gram double hash per perm** | `src/reranker/heuristics/lsh.py:43-47` | Low | SHA-256 per n-gram per permutation (128× per n-gram). Fine for small texts. |
| **`_fit_bilinear` refits LogisticRegression from scratch** | `src/reranker/strategies/binary_reranker.py:68-100` | Low | No incremental/online fitting. |
| **No vectorized batch scoring in SPLADE** | `src/reranker/strategies/splade.py:84-89` | Low | Iterates doc-by-doc in Python loop. Fine for sparse dicts. |

## Maintenance Issues

| Issue | File:Line | Severity | Notes |
|---|---|---|---|
| **Pre-commit ruff excludes `tests/` and `data/_expanded/`** | `.pre-commit-config.yaml:29-33` | Medium | Tests and expanded seeds file excluded from ruff linting. Style drift possible in test code. |
| **Bandit skip B101 (assert), B311 (random)** | `pyproject.toml:114` | Low | Intentional — asserts used in tests, random used for seeds. |
| **19 bare `except Exception` handlers** | Throughout `src/` | Medium | See Security section. Many should either log or catch specific exceptions. |
| **Coverage floor at 85%** | `pyproject.toml:103` | Low | Industry standard; many integration/e2e tests excluded from coverage by default marker filters. |
| **Ruff selected rules limited** | `pyproject.toml:86` | Low | Only E, F, I, B, UP. Missing: SIM (simplify), PL (pylint), RUF (ruff-specific), PERF (performance). |
| **`# type: ignore` scattered** | Various | Low | ~10 `type: ignore` comments across `src/`. Needed for optional dependency typing. |
| **No mypy on tests/scripts/benchmarks/notebooks** | `.pre-commit-config.yaml:58` | Low | Excluded from mypy checks. Type errors in test code go undetected. |
| **Ruff + mypy pass cleanly** | — | — | Zero ruff violations in `src/`, zero mypy errors in 73 source files. |
