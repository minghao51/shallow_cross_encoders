# External Integrations

## LLM & AI Services

### OpenRouter (via httpx)
- **Purpose**: Synthetic training data generation when labeled data unavailable
- **Client**: src/reranker/data/client.py (OpenRouterClient)
- **API**: https://openrouter.ai/api/v1
- **Library**: httpx (raw API calls)
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

### LiteLLM
- **Purpose**: Active distillation teacher model completions
- **Client**: src/reranker/data/litellm_client.py (LiteLLMClient)
- **Library**: litellm
- **Usage**: Teacher labeling in active distillation loop
- **Auth**: LITELLM_API_KEY (environment variable)
- **Default Model**: openrouter/openai/gpt-4o-mini
- **Provider Key**: "litellm" in config

### Google GenAI (Gemini)
- **Purpose**: Alternative LLM provider for synthetic data generation and active distillation
- **Client**: src/reranker/data/genai_client.py (GenAIClient)
- **SDK**: google-genai (client.models.generate_content)
- **JSON Mode**: response_mime_type="application/json" via GenerateContentConfig
- **Auth**: GOOGLE_GENAI_API_KEY (environment variable)
- **Default Model**: gemini-2.5-flash
- **Provider Key**: "genai" in config

### Unified LLM Client
- **Protocol**: LLMClient (runtime_checkable, .enabled + .complete_json)
- **Type**: LLMClientType = OpenRouterClient | LiteLLMClient | GenAIClient
- **Factory**: create_llm_client(provider) in src/reranker/data/__init__.py
- **Config**: llm_provider field on each consumer's settings (synthetic_data.llm_provider, active_distillation.llm_provider)
- **All three clients share the same complete_json(prompt) -> (dict, metadata) interface

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
- **Status**: Core dependency (included in base install)

### Model2Vec
- **Purpose**: Static embeddings for CPU-native inference
- **Usage**:
  - Hybrid fusion reranker
  - Embedding generation for documents/queries
- **Status**: Core dependency (included in base install)
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
