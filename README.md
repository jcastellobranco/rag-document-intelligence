# RAG Document Intelligence

Document ingestion and tenant-scoped retrieval with chunk-level citations, hybrid ranking, and a pluggable embedding provider.

## What works

- Text ingestion and PDF extraction in the original SQLite FTS5 demo API (`src/rag_intelligence/service.py`).
- Separate PostgreSQL + pgvector API (`src/rag_intelligence/vector_api.py`) with text/PDF ingestion and hybrid search via FTS + cosine similarity + Reciprocal Rank Fusion (RRF).
- 384-dimensional sentence-transformers embedding adapter (default `all-MiniLM-L6-v2`), lazily loaded; model weights may require a download.
- PostgreSQL RLS with `app.tenant_id` set per transaction, plus a server-bound tenant ID and API key.
- Tests for security, API contract, deterministic embedding adapter and pgvector integration, run by GitHub Actions.

## Run

See [Vector HTTP API](docs/VECTOR_HTTP_API.md), [Vector architecture](docs/VECTOR_ARCHITECTURE.md), and [Embedding provider](docs/EMBEDDINGS.md). Local database: `docker compose -f docker-compose.vector.yml up -d`.

## Integration

The sibling `ai-inference-platform` calls `POST /v1/search` on the vector API and passes returned evidence to an extractive response or optional LLM provider. Results and citations can be submitted to `llm-evaluation-observability`.

## Local Prometheus metrics

The vector API exposes `GET /metrics/` on its **private** service port for
the ATLAS Prometheus scraper. It tracks search latency, retrieved-hit counts,
embedding encode latency, ingested chunks and HTTP request status by a fixed
operation name. Metrics do not label tenant IDs, questions, document names,
chunk IDs or raw paths. Prometheus discovers the service at `rag:8000` in
the sibling `ai-inference-platform` Compose overlay. The
`llm-evaluation-observability` repository owns the Grafana dashboard.

Use `make up-observability` in the sibling inference checkout to start
RAG with Prometheus, Grafana, Loki and per-container logs. The metrics endpoint
is not authenticated; keep the RAG container off public ports.

## Boundaries

The vector API is intentionally *single-tenant per deployment*. A tenant header is not authentication; callers must bind the authenticated tenant to the server configuration. No GraphRAG or knowledge graph is implemented. A successful CI run validates controlled pgvector tests, not real-model retrieval quality or production deployment. The original SQLite endpoint remains available as a separate demo.

This is a portfolio/reference system, **not production-ready**.
