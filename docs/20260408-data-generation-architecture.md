# Data Generation Architecture

The synthetic data stack now uses small internal packages instead of a pair of monolithic files.

## Synthetic Generator

`src/reranker/data/synth/_generator.py` remains the public facade. It delegates to
`src/reranker/data/synth/generator/`:

- `core.py` handles schema validation, chunking, teacher-mode orchestration, and cost logging.
- `pairs.py`, `preferences.py`, and `contradictions.py` own the core dataset builders.
- `enhanced.py` owns hard negatives, listwise preferences, and query expansions.
- `artifacts.py` owns manifest creation, label summaries, and chunked JSONL materialization.

Public list-returning methods now have streaming counterparts such as `iter_pairs`,
`iter_preferences`, and `iter_contradictions`. `materialize_all()` consumes those streaming
primitives and writes JSONL outputs in chunks controlled by `RERANKER_STREAM_CHUNK_SIZE`.

## Expanded Offline Datasets

`src/reranker/data/expanded.py` is now a compatibility module that re-exports from
`src/reranker/data/_expanded/`:

- `seeds.py` contains the static domain seed corpus.
- `pairs.py`, `preferences.py`, and `contradictions.py` contain the dataset-specific builders.
- `helpers.py` and `types.py` provide shared sampling utilities and typed record shapes.

The expanded dataset module also exposes iterator-based APIs:

- `iter_expanded_pairs`
- `iter_expanded_preferences`
- `iter_expanded_contradictions`

The original `generate_expanded_*` functions remain import-compatible and simply materialize the
iterator outputs into lists.

## Extension Points

- Add or revise offline seed content in `src/reranker/data/_expanded/seeds.py`.
- Add teacher prompt variants in `src/reranker/data/synth/_prompts.py`.
- Extend synthetic record schemas in `src/reranker/data/synth/_models.py`.
- Prefer adding new generation behaviors as helper modules under
  `src/reranker/data/synth/generator/` rather than growing the facade class.
