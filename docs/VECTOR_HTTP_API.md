# Pgvector-backed HTTP service

The original `service.py` demonstrates SQLite FTS retrieval. **The new `vector_api.py` is the separate PostgreSQL+pgvector HTTP implementation** with identical ingest/search routes and JSON citation fields compatible with the inference platform.

## Local setup

Run `docker compose -f docker-compose.vector.yml up -d` to prepare the database and apply migrations. Set the following variables for the API:

- `RAG_DATABASE_URL=postgresql+psycopg://rag_runtime:local-rag-development-only@localhost:55432/rag`
- `RAG_TENANT_ID=11111111-1111-4111-8111-111111111111`
- `RAG_API_KEY=<a separately generated secret>`

Then `pip install -e '.[vector,embeddings]'` and
`uvicorn rag_intelligence.vector_api:create_app --factory --host 127.0.0.1 --port 8010`.

The first real sentence-transformers invocation may download model weights and should be run in an environment with approved access or pre-cached model weights.

```bash
curl -X POST http://localhost:8010/v1/documents \
  -H "x-api-key: $RAG_API_KEY" -H "x-tenant-id: $RAG_TENANT_ID" \
  -H "content-type: application/json" \
  -d '{"source":"demo.md","text":"PostgreSQL supports vector similarity search"}'
```

A second `POST /v1/search` with JSON `{"question":"vector search","limit":5}` returns citations. API authorization is bound to the single configured UUID; database requests additionally enforce RLS. **Do not grant the runtime DB account table ownership or BYPASSRLS.**

CI uses a fake model against real pgvector to verify the HTTP/SQL contract without downloading model weights. The actual model execution remains a separate smoke test; search quality also requires a representative dataset.
