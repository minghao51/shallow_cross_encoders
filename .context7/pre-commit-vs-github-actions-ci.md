# Pre-commit vs GitHub Actions CI: Best Practices

## Overview

Both pre-commit hooks and GitHub Actions CI serve to enforce code quality, but they operate at different stages of the development workflow and serve different purposes.

---

## Pre-commit Hooks

### What It Is
A framework for managing and maintaining multi-language pre-commit hooks. Runs locally before every commit.

### When to Use Pre-commit
- **Fast feedback loops** - Catches issues before they're committed
- **Style and formatting** - Linting, formatting, trailing whitespace fixes
- **Language-agnostic checks** - Works for any language without project dependencies
- **Prevents bad commits** - Ensures commits are always clean
- **Local validation** - No network dependency, runs offline

### Strengths
1. **Immediate feedback** - Developer sees issues before commit
2. **No revert needed** - Fix issues before push
3. **Saves CI resources** - Bad code never reaches CI
4. **Cross-platform** - Works on all developer machines
5. **No waiting time** - No queue, no CI congestion
6. **Selective file checking** - Only checks staged files

### Limitations
1. **Can be bypassed** - `git commit --no-verify` skips hooks
2. **Install burden** - Must run `pre-commit install` on each machine
3. **Limited scope** - Not suitable for slow/large checks
4. **No visibility** - Results don't persist or get shared

### Best Practices
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-merge-conflict

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## GitHub Actions CI

### What It Is
Automated workflows that run on GitHub infrastructure when events occur (push, PR, etc.)

### When to Use CI
- **Cross-platform testing** - Linux, macOS, Windows matrix
- **Slow/comprehensive checks** - Full test suites, coverage reports
- **Security scanning** - pip-audit, dependency vulnerabilities
- **Deployment** - Release, deploy, publish packages
- **Visibility** - Results visible to entire team
- **Audit trail** - Historical record of build status

### Strengths
1. **Centralized enforcement** - Can't be bypassed by individual devs
2. **Comprehensive** - Can run full test matrix
3. **Artifact storage** - Test results, coverage, builds persist
4. **Security scanning** - Runs pip-audit, security checks
5. **Cross-platform** - Tests on multiple OSes/versions
6. **Team visibility** - All see results in PR

### Limitations
1. **Runs after push** - Issues already in repository
2. **Requires fix commit/amend** - Can't prevent initial bad commit
3. **Wait time** - Queue, runner availability, network latency
4. **Resource consumption** - Uses GitHub compute minutes
5. **Friction** - Blocks merge, causes PR backpressure

### Best Practices
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install pre-commit
      - run: pre-commit run --all-files

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12', '3.13']
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e .[dev]
      - run: pytest --cov=reranker --cov-report=xml

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - run: pip install pip-audit
      - run: pip-audit --require-hashes --recursive .
```

---

## Decision Matrix: When to Use What

| Check Type | Pre-commit | CI | Reason |
|------------|------------|-----|--------|
| Ruff/mypy linting | ✅ | ✅ | Fast enough for pre-commit, but CI catches skipped hooks |
| Formatting (black, ruff format) | ✅ | ❌ | Pre-commit auto-fixes before commit |
| Unit tests | ❌ | ✅ | Too slow for pre-commit |
| Integration tests | ❌ | ✅ | May need services (DB, etc.) |
| Security scans (pip-audit) | ❌ | ✅ | Requires up-to-date vulnerability DB |
| Coverage reports | ❌ | ✅ | Needs artifact storage |
| Cross-platform testing | ❌ | ✅ | Requires multiple runners |
| Deployments | ❌ | ✅ | Needs credentials/secrets |
| Large/slow checks | ❌ | ✅ | Would frustrate developers |

---

## Recommended Architecture

### 1. Pre-commit as First Gate
```
git commit → pre-commit hooks run →
  - If FAIL: commit blocked, developer fixes
  - If PASS: commit succeeds
```

**Use for:** Fast, auto-fixable, stylistic checks

### 2. CI as Verification Layer
```
git push → CI runs →
  - If FAIL: PR blocked or revert required
  - If PASS: Ready to merge
```

**Use for:** Everything pre-commit can't handle

### 3. Ideal Workflow
```
Developer writes code
    ↓
git add + git commit
    ↓
Pre-commit runs (seconds)
  - Auto-format fixes
  - Lint checks
  - Type checks
    ↓
If fails → Developer fixes locally (no push)
If passes → Commit succeeds
    ↓
git push → PR created
    ↓
CI runs (minutes)
  - Full test suite
  - Security scans
  - Coverage reports
    ↓
If fails → Developer pushes fix commit
If passes → Ready to merge
```

---

## Key Insights

### Pre-commit Advantages
1. **Prevents issues at source** - No need to revert/amend
2. **Faster iteration** - No CI queue, no waiting
3. **Clean commits** - History always valid
4. **Developer autonomy** - Fix at your own pace

### CI Advantages
1. **Cannot be bypassed** - Even `git commit --no-verify` still goes through CI on push
2. **Comprehensive** - Full test matrix, security scans
3. **Team visibility** - Everyone sees results
4. **Audit trail** - Historical record

### The Critical Difference
> **Pre-commit prevents bad code from entering the repo.**
> **CI verifies what already landed.**

If your team frequently bypasses pre-commit, CI is essential. But CI alone means you're fixing issues after they've been committed, requiring revert or amend workflows.

---

## Conclusion

**Use pre-commit for:** Fast, auto-fixable checks that should never fail in a committed state (lint, format, type check, basic validation).

**Use CI for:** Comprehensive checks that pre-commit can't do (slow tests, security scans, coverage, cross-platform, deployments).

**Best practice:** Use both. Pre-commit as the first gate to keep commits clean. CI as the second gate to catch anything that slips through or requires comprehensive verification.

This way:
- Developers get fast feedback
- Commits are always valid
- CI doesn't waste resources on obvious issues
- Security scans run on every change
- Team has visibility into build status
