# Document Intelligence: architecture and boundaries

The HTTP API authenticates a configured service API key, enforces a single server-configured tenant, and accepts plain text or PDF uploads. Extracted PDF text is chunked with overlap and persisted in SQLite. FTS5 provides lexical BM25 ranking and chunk-level citations.

This implementation is **retrieval**, not yet full generative RAG: it does not produce LLM answers or embeddings. SQLite is a local demonstration backend, not a recommended shared distributed multi-tenant database.

## Security and reliability

- Uploads are limited to 10 MiB, 100 pages and 200k extracted characters.
- PDFs with no extractable text return 422; OCR is deliberately out of scope.
- Auth is required for document ingestion and search.
- Tenant comes from server configuration; a request header cannot select another tenant.
- Search queries use SQLite parameter binding; FTS match terms are quoted.
- PDF parsing remains a potentially expensive operation. Production deployment needs process isolation, CPU/time limits and malware scanning.

## Next architectural milestone

Replace the local SQLite store with PostgreSQL + pgvector, enforce tenant RLS, store source metadata, use an injectable embedding provider, and implement grounded answer generation with an explicit refusal when citations are insufficient.

## Local run

```bash
pip install -e '.[dev,pdf]'
export RAG_API_KEY=local-demo-secret
uvicorn rag_intelligence.service:create_app --factory --reload
pytest -q
```
