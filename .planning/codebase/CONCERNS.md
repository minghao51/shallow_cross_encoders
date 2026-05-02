# Codebase Concerns

**Analysis Date:** 2026-04-23

## Tech Debt

**Large files (>500 lines):**
- Issue: Several files exceed recommended size limits, making maintenance and comprehension difficult
- Files: `src/reranker/benchmark.py` (1106 lines), `src/reranker/data/_expanded/seeds.py` (706 lines), `src/reranker/strategies/hybrid.py` (504 lines)
- Impact: Harder to navigate, test, and modify; increased cognitive load for developers
- Fix approach: Extract helper modules, use composition, split by responsibility

**Scripts directory complexity:**
- Issue: Multiple large scripts in root `scripts/` directory with overlapping concerns
- Files: `scripts/distill_ensemble_to_hybrid.py` (551 lines), `scripts/benchmark_real_data.py` (442 lines), `scripts/run_beir_benchmark.py` (470 lines)
- Impact: Code duplication, inconsistent patterns, difficult to maintain
- Fix approach: Create shared utility modules, extract common benchmarking logic

## Known Bugs

**None identified** during this analysis

## Security Considerations

**API key management:**
- Risk: API keys referenced in code but fetched from settings/environment
- Files: `src/reranker/data/litellm_client.py:27-51`
- Current mitigation: Uses `get_settings().active_distillation.litellm_api_key` and environment-based configuration
- Recommendations: Validate that secrets are never committed; ensure .env files are in .gitignore

**Bare exception handlers:**
- Risk: 38 bare exception handlers (`except:` or `except Exception`) could mask errors
- Files: Across `src/` (specific files not traced in this analysis)
- Current mitigation: Limited
- Recommendations: Audit bare excepts, add specific exception types, implement logging

## Performance Bottlenecks

**No sleep/time-based delays detected** in main source code

## Fragile Areas

**Synthetic data generation:**
- Files: `src/reranker/data/synth/` (multiple modules)
- Why fragile: Complex generation pipeline, depends on external APIs (OpenRouter)
- Safe modification: Add comprehensive unit tests, mock external dependencies
- Test coverage: Needs verification

**Benchmark suite:**
- Files: `src/reranker/benchmark.py`, multiple scripts in `scripts/`
- Why fragile: Large monolithic file, multiple dependencies, external data sources
- Safe modification: Extract strategy-specific benchmarks, parameterize configurations
- Test coverage: Needs verification

## Scaling Limits

**Not identified** during this analysis

## Dependencies at Risk

**None identified** during this analysis

## Missing Critical Features

**None identified** during this analysis

## Test Coverage Gaps

**Large files with unverified test coverage:**
- What's not tested: Large utility and benchmark files may lack comprehensive coverage
- Files: `src/reranker/benchmark.py`, `src/reranker/strategies/hybrid.py`
- Risk: Regression bugs in complex logic
- Priority: Medium (372 test functions exist overall)

**Error handling paths:**
- What's not tested: Bare exception handlers' error scenarios
- Files: Files with `except:` and `except Exception` blocks
- Risk: Errors being silently ignored
- Priority: High

---

*Concerns audit: 2026-04-23*
