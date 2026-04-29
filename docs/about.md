# About

**shallow-cross-encoders** is a CPU-native reranking and consistency pipeline that combines static embeddings, lexical signals, and shallow models for fast, effective document ranking.

## Architecture

The library is organized around a protocols-first design:

- **Protocols** (`reranker.protocols`) define the `BaseReranker` and `RankedDoc` interfaces
- **Strategies** (`reranker.strategies`) implement concrete ranking algorithms
- **Data** (`reranker.data`) handles loading, sampling, and synthetic generation
- **Evaluation** (`reranker.eval`) provides metrics and benchmarking utilities

## License

See the [repository](https://github.com/minghao/shallow_cross_encoders) for license information.
