# External Integrations

## LLM & AI Services

### OpenRouter (via LiteLLM)
- **Purpose**: Synthetic training data generation when labeled data unavailable
- **Client**: src/reranker/data/client.py (OpenRouterClient)
- **API**: https://openrouter.ai/api/v1
- **Library**: litellm
- **Usage**:
  - Generate query-document pairs
  - Generate preference rankings
  - Generate contradiction examples
  - Generate expanded query variations
- **Auth**: OPENROUTER_API_KEY (environment variable)
- **Default Model**: openai/gpt-4o-mini
- **Rate Limiting**: Batch processing (configurable batch size, max workers)
- **Fallback**: None (required for teacher mode)
- **Mock Support**: Yes (pytest mocks for testing)

## Model Providers

### FlashRank
- **Purpose**: Cross-encoder models for teacher distillation and high-quality reranking
- **Format**: ONNX runtime
- **Models**:
  - ms-marco-MiniLM-L-6-v2 (slower, higher quality)
  - ms-marco-TinyBERT-L-2-v2 (faster, good quality)
- **Usage**:
  - Teacher model for distillation
  - Fallback strategy in cascade reranker
  - Benchmarking baseline
- **Status**: Optional dependency (--extra flashrank)

### Model2Vec
- **Purpose**: Static embeddings for CPU-native inference
- **Usage**:
  - Hybrid fusion reranker
  - Embedding generation for documents/queries
- **Status**: Optional dependency (--extra runtime)
- **Fallback**: Uses SentenceTransformers if unavailable

### SentenceTransformers
- **Purpose**: PyTorch-based sentence embeddings
- **Usage**:
  - Alternative embedding backend
  - Benchmarking comparison
  - Late interaction models (ColBERT)
- **Models**:
  - all-MiniLM-L-6-v2
  - BAAI/bge-base-en-v1.5 (for late interaction)
- **Status**: Optional dependency (--extra sentence-transformers)

### HuggingFace (implicit)
- **Purpose**: Model hosting and download
- **Access**: Via FlashRank, Model2Vec, SentenceTransformers libraries
- **Models Downloaded**:
  - minishlab/potion-base-8M (static embeddings)
  - ms-marco-* cross-encoders
  - all-MiniLM-L-6-v2

## Databases & Storage

### Local File System
- **Storage Type**: JSON and JSONL files
- **Locations**:
  - data/raw/: Raw datasets (manifest.json, seeds, etc.)
  - data/processed/: Processed labels and features
  - data/logs/: API costs and metadata (api_costs.jsonl)
- **No Database**: All data file-based
- **Caching**: src/reranker/data/ensemble_cache.py (hash-based label cache)

### External Datasets
- **BEIR**: Information retrieval benchmark datasets
  - nfcorpus, scifact, fluent-legal
  - Download via scripts/download_beir.py
  - Local cache: data/beir/

## HTTP & Network

### HTTP Client
- **Library**: httpx (async HTTP)
- **Features**:
  - Connection pooling (20 keepalive, 100 max)
  - Timeout handling
  - Retry logic (via tenacity)
- **Usage**: OpenRouter API calls
- **Test Client**: Mock support for testing

## Auth & Security

### Environment Variables
- **OPENROUTER_API_KEY**: LLM API authentication
- **LITELLM_API_KEY**: Alternative LLM auth
- No other external auth required

### Secrets Management
- **Storage**: .env file (gitignored)
- **Example**: .env.example provided
- **No Vault/Secrets Service**: File-based only

## Third-Party Services

### None
- No webhooks
- No message queues (Kafka, RabbitMQ, etc.)
- No monitoring (Prometheus, Datadog, etc.)
- No logging services (ELK, etc.)
- No CDNs or object storage (S3, GCS, etc.)

## Integration Patterns

### Teacher-Student Distillation
- **FlashRank** (teacher) → Local models (Hybrid, Distilled)
- **OpenRouter** (teacher) → Synthetic labels

### Cascade Strategy
- Fast models (Hybrid, Distilled) → FlashRank fallback
- Confidence-based routing
- Stats collection for observability

### Synthetic Data Generation
- OpenRouter LLM generates labeled examples
- Multi-stage: pairs → preferences → contradictions
- Cost tracking in data/logs/api_costs.jsonl

## External Dependencies Summary

| Category | Services | Auth Required | Optional |
|----------|----------|---------------|----------|
| LLM APIs | OpenRouter | Yes | No |
| Models | FlashRank, Model2Vec, SentenceTransformers | No | Yes |
| Datasets | BEIR | No | Yes |
| Storage | Local files only | No | No |
| Network | HTTP only (OpenRouter) | Yes | No |
| Message Queues | None | - | - |
| Monitoring | None | - | - |
