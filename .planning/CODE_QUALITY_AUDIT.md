# Code Quality & Style Audit Report

**Date:** 2026-05-24  
**Scope:** `src/reranker/` (all subdirectories)  
**Files reviewed:** 75+ Python files across `strategies/`, `data/`, `eval/`, `cli/`, `heuristics/`, and top-level modules

---

## Summary

| Category | High | Medium | Low | Total |
|---|---|---|---|---|
| 1. Inconsistent patterns | 3 | 1 | 1 | 5 |
| 2. Missing docstrings | 7 | 6 | 6 | 19 |
| 3. Type hint issues | 2 | 3 | 2 | 7 |
| 4. Import styles | 0 | 0 | 0 | 0 |
| 5. Dead code/logic | 2 | 1 | 0 | 3 |
| 6. Error handling | 0 | 2 | 1 | 3 |
| 7. Naming violations | 0 | 0 | 0 | 0 |
| 8. Functions too long | 3 | 12 | 5 | 20 |
| 9. Duplicated code | 2 | 3 | 0 | 5 |
| 10. Magic numbers | 0 | 5 | 6 | 11 |
| **Total** | **19** | **33** | **21** | **73** |

---

## Top 5 Highest-Impact Fixes (by effort-to-impact ratio)

1. **Extract `evaluate_strategy` into a dispatch table** (`eval/runner.py`) — eliminates 296 lines of duplication
2. **Fix bare `list` return types** on `rank_docs`, `BM25Engine.rerank`, `FlashRankEnsemble.rerank`
3. **Remove 7 duplicate docstrings** in `data/_expanded/` and `data/synth/generator/`
4. **Fix dead branches** in `config.py:340` and `data/client.py:148-153`
5. **Extract shared teacher-batch-retry pattern** in `data/synth/generator/`

---

## Category 1: Inconsistent Code Patterns

### 1.1 `RankedDoc` return type inconsistency

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/utils.py:251` | **high** | `rank_docs()` returns bare `list` instead of `list[RankedDoc]` — this is the central helper used by all strategies |
| `src/reranker/lexical.py:171` | **high** | `BM25Engine.rerank()` returns bare `list` instead of `list[RankedDoc]` |
| `src/reranker/strategies/flashrank_ensemble.py:57` | **high** | `FlashRankEnsemble.rerank()` returns `list[Any]` instead of `list[RankedDoc]` |

**Fix:** Change all three to `list[RankedDoc]`.

### 1.2 Inconsistent "not fitted" error types

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/strategies/hybrid.py:402` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/distilled.py:234` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/distilled.py:275` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/binary_reranker.py:149` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/binary_reranker.py:197` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/splade.py:66` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/splade.py:103` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/late_interaction.py:144` | **medium** | raises `RuntimeError` |
| `src/reranker/strategies/late_interaction.py:188` | **medium** | raises `RuntimeError` |

All use `RuntimeError` consistently — but should be a custom exception type (e.g., `NotFittedError`) for better catchability.

**Fix:** Create a `NotFittedError` in `protocols.py` and use it everywhere.

### 1.3 EmbeddingCache stats uses ints instead of bools

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/embedding_cache.py:91-97` | **low** | `stats()` returns `{"enabled": 1, ...}` using `1`/`0` instead of `True`/`False` |

**Fix:** Use boolean for `enabled` key.

---

## Category 2: Missing or Inconsistent Docstrings

### 2.1 Missing module-level docstrings

| File | Severity |
|---|---|
| `src/reranker/utils.py` | **medium** |
| `src/reranker/strategies/multi.py` | **medium** |
| `src/reranker/strategies/meta_router.py` | **medium** |
| `src/reranker/strategies/splade.py` | **medium** |
| `src/reranker/embedding_cache.py` | **medium** |

### 2.2 Missing class docstrings

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/embedding_cache.py:16` | **medium** | `EmbeddingCache` class missing docstring |

### 2.3 Missing function docstrings

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/quantization.py:125` | **low** | `quantize_int8` missing docstring |
| `src/reranker/quantization.py:141` | **low** | `dequantize_int8` missing docstring |
| `src/reranker/quantization.py:148` | **low** | `quantize_float16` missing docstring |
| `src/reranker/quantization.py:160` | **low** | `dequantize_float16` missing docstring |
| `src/reranker/eval/runner.py:51` | **low** | `_mean` missing docstring |
| `src/reranker/eval/runner.py:55` | **low** | `_hybrid_model_path` missing docstring |
| `src/reranker/data/synth/generator/core.py:95` | **low** | `normalize_generated_value` missing docstring |
| `src/reranker/data/synth/generator/core.py:207` | **low** | `collect_records` missing docstring |

### 2.4 Duplicate docstrings (string appears twice in same function)

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/data/synth/generator/contradictions.py:175` | **high** | Second `"""Yield contradiction and control records."""` after full docstring on line 163 |
| `src/reranker/data/_expanded/pairs.py:29` | **high** | Second docstring after full docstring on line 19 |
| `src/reranker/data/_expanded/pairs.py:87` | **high** | Second docstring after full docstring on line 78 |
| `src/reranker/data/_expanded/preferences.py:62` | **high** | Second docstring after full docstring on line 51 |
| `src/reranker/data/_expanded/preferences.py:166` | **high** | Second docstring after full docstring on line 157 |
| `src/reranker/data/_expanded/contradictions.py:225` | **high** | Second docstring after full docstring on line 214 |
| `src/reranker/data/_expanded/contradictions.py:275` | **high** | Second docstring after full docstring on line 264 |

**Fix:** Remove the duplicate (shorter) docstring in each case.

---

## Category 3: Inconsistent Type Hint Usage

| File:Line | Severity | Issue | Fix |
|---|---|---|---|
| `src/reranker/utils.py:251` | **high** | `rank_docs` returns bare `list` | Use `list[RankedDoc]` |
| `src/reranker/lexical.py:171` | **high** | `rerank` returns bare `list` | Use `list[RankedDoc]` |
| `src/reranker/data/ensemble_cache.py:45` | **medium** | `_convert_tuples_to_lists(self, labels: dict)` untyped dict | Use `dict[tuple, Any]` |
| `src/reranker/data/ensemble_cache.py:56` | **medium** | `_convert_lists_to_tuples(self, labels: dict)` untyped dict | Use `dict[str, Any]` |
| `src/reranker/data/ensemble_cache.py:77` | **medium** | `load_or_generate` returns bare `dict` | Use `dict[Any, Any]` |
| `src/reranker/data/custom_beir.py:15` | **low** | `load_custom_beir` returns bare `dict` | Use `dict[str, Any]` |
| `src/reranker/data/litellm_client.py:98-99` | **low** | JSON decode error silently swallowed, returns `{"raw": content}` | Consider logging or raising |

---

## Category 4: Import Styles

All files consistently use absolute imports (`from reranker.config import ...`). **No issues found.**

---

## Category 5: Dead Code / Unused Imports / Dead Logic

| File:Line | Severity | Issue | Fix |
|---|---|---|---|
| `src/reranker/config.py:340` | **high** | `{k: Path(v) if k != "api_cost_log" else Path(v) ...}` — both branches identical, condition is dead | Simplify to `{k: Path(v) for k, v in overrides.items()}` |
| `src/reranker/data/client.py:148-153` | **high** | Both branches of the `if exc.response.status_code == 400: continue` do `continue` — condition is dead | Log 400 errors differently or remove the condition |
| `src/reranker/strategies/flashrank_ensemble.py:9` | **medium** | Module-level import of `prepare_benchmark_data_with_hard_negatives` — heavy import only used in `HardNegativeFlashRankEnsemble` | Move import to method body |

---

## Category 6: Inconsistent Error Handling

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/data/litellm_client.py:98-99` | **medium** | `json.JSONDecodeError` silently returns `{"raw": content}` instead of raising — contradicts `OpenRouterClient._extract_json_or_raise` which raises `ValueError` |
| `src/reranker/data/client.py:148-153` | **medium** | Both `httpx.HTTPStatusError` branches `continue` — status code 400 check is meaningless |
| `src/reranker/data/beir_loader.py:46` vs `data/custom_beir.py:52` | **low** | `load_beir_simple` raises `ImportError` for missing beir; `load_custom_beir` wraps in `ValueError` — inconsistent error wrapping for similar "missing dep/file" cases |

---

## Category 7: Naming Convention Violations

**No snake_case violations found.** All names are consistent.

---

## Category 8: Functions/Methods Too Long (>50 lines)

| File:Line | Severity | Lines | Function |
|---|---|---|---|
| `src/reranker/eval/runner.py:129` | **high** | 296 | `evaluate_strategy` — massive if/elif chain with duplicated logic |
| `src/reranker/data/beir_loader.py:101` | **high** | 140 | `load_beir_comprehensive` |
| `src/reranker/data/hard_negative_sampler.py:87` | **high** | 135 | `prepare_benchmark_data_with_hard_negatives` |
| `src/reranker/data/synth/generator/enhanced.py:69` | **medium** | 123 | `iter_hard_negatives` |
| `src/reranker/data/synth/generator/contradictions.py:158` | **medium** | 117 | `iter_contradictions` |
| `src/reranker/data/beir_loader.py:19` | **medium** | 80 | `load_beir_simple` |
| `src/reranker/data/synth/generator/enhanced.py:194` | **medium** | 96 | `iter_listwise_preferences` |
| `src/reranker/data/synth/generator/enhanced.py:292` | **medium** | 86 | `iter_query_expansions` |
| `src/reranker/data/synth/generator/preferences.py:126` | **medium** | 88 | `iter_preferences` |
| `src/reranker/data/client.py:103` | **medium** | 76 | `complete_json` |
| `src/reranker/eval/viz.py:177` | **medium** | 89 | `generate_comparison_table` |
| `src/reranker/eval/viz.py:101` | **medium** | 74 | `plot_radar` |
| `src/reranker/eval/viz.py:31` | **medium** | 68 | `plot_pareto_frontier` |
| `src/reranker/eval/benchmark_utils.py:16` | **medium** | 77 | `evaluate_reranker_on_rows` |
| `src/reranker/strategies/hybrid.py:131` | **medium** | 65 | `_build_features` |
| `src/reranker/cli/benchmark.py:86` | **low** | 81 | `benchmark_compare` |
| `src/reranker/data/active_distill.py:237` | **low** | 60 | `ActiveDistiller.run` |
| `src/reranker/strategies/consistency.py:282` | **low** | 89 | `ConsistencyEngine.check` |
| `src/reranker/strategies/consistency.py:396` | **low** | 66 | `ConsistencyEngine.diagnose_misses` |
| `src/reranker/eval/runner.py:76` | **low** | 51 | `_metrics_for_rows` |

**Fix:** Extract helper functions. `evaluate_strategy` is the worst offender — should be a strategy dispatch table.

---

## Category 9: Duplicated Code

| File:Line | Severity | Issue |
|---|---|---|
| `src/reranker/eval/runner.py:129-424` | **high** | `evaluate_strategy` has ~6 near-identical blocks for data loading -> partitioning -> training -> evaluating. Only the strategy-specific parts differ. |
| `src/reranker/cli/train.py` (multiple) | **high** | Every train command repeats: `_load_rows` -> `_partition_train_rows` -> train -> save -> `_print_train_summary`. Should be a shared helper. |
| `src/reranker/data/synth/generator/enhanced.py:69-377` | **medium** | `iter_hard_negatives`, `iter_listwise_preferences`, and `iter_query_expansions` share the same teacher-batch-with-retry pattern. Extract a shared `_iter_teacher_batches()` helper. |
| `src/reranker/data/active_distill.py:298-359` | **medium** | `_derive_preferences` duplicates preference-derivation logic also present in `data/synth/generator/preferences.py:166-189` |
| `src/reranker/data/synth/generator/contradictions.py:88-155` | **medium** | `teacher_contradiction_records` duplicates the binary-split-on-failure retry pattern from `pairs.py:73-139` and `preferences.py:67-123` |

**Fix:** Extract shared patterns into helper functions in `core.py` or a new `retry.py`.

---

## Category 10: Magic Numbers / Hardcoded Strings

| File:Line | Severity | Magic Value | Fix |
|---|---|---|---|
| `src/reranker/data/client.py:126` | **medium** | `"https://local.shallow-cross-encoders"` | Move to config or constant |
| `src/reranker/data/client.py:143` | **medium** | `0.2` temperature | Move to `OpenRouterSettings` |
| `src/reranker/data/active_distill.py:287` | **medium** | `0.0004` cost per judgment | Move to `RoiSettings` |
| `src/reranker/data/litellm_client.py:72` | **medium** | `0.2` temperature | Move to `ActiveDistillationSettings` |
| `src/reranker/data/hard_negative_sampler.py:146` | **low** | `500` text truncation | Extract constant |
| `src/reranker/data/hard_negative_sampler.py:207` | **low** | `800` FlashRank truncation | Extract constant |
| `src/reranker/strategies/distilled.py:90` | **medium** | `7` feature dimension hardcoded | Extract as `_PAIRWISE_FEATURE_DIM = 7` |
| `src/reranker/strategies/distilled.py:153` | **medium** | `32` batch size | Move to config |
| `src/reranker/strategies/binary_reranker.py:55` | **low** | `10_000` cache size | Move to config |
| `src/reranker/strategies/binary_reranker.py:81-82` | **low** | `1.0` C, `500` max_iter | Already in `DistilledSettings` but not `BinaryRerankerSettings` |
| `src/reranker/strategies/consistency.py:304` | **low** | `0.1 * score_scale` threshold | Extract constant |
| `src/reranker/lexical.py:66-67` | **low** | `1.5` k1, `0.75` b | Extract as `BM25_DEFAULT_K1`, `BM25_DEFAULT_B` |
| `src/reranker/eval/metrics.py:15` | **low** | `2.0` default threshold | Already configurable per-call, acceptable |
| `src/reranker/logging_config.py:73` | **low** | Hardcoded logger names | Acceptable for noise suppression |
