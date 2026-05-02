## 1. Workflow
- **Analyze First:** Read relevant files before proposing solutions. Never hallucinate.
- **Approve Changes:** Present a plan for approval before modifying code.
- **Minimal Scope:** Change as little code as possible. No new abstractions.
- **Python Execution:** Always `uv run <command>`. Never `python`. Sync with `uv sync`.
- **Docs:** Update `ARCHITECTURE.md` if structure changes. Markdown files follow `YYYYMMDD-filename.md`.

## 2. Output Style
- High-level summaries only.
- No speculation about code you haven't read.

## 3. Project Context
- **Architecture & modules** → @.planning/codebase/ARCHITECTURE.md
- **Stack & dependencies** → @.planning/codebase/STACK.md
- **Coding conventions** → @.planning/codebase/CONVENTIONS.md
- **External integrations** → @.planning/codebase/INTEGRATIONS.md
- **Testing practices** → @.planning/codebase/TESTING.md
- **Known concerns & tech debt** → @.planning/codebase/CONCERNS.md
- **Directory structure** → @.planning/codebase/STRUCTURE.md
