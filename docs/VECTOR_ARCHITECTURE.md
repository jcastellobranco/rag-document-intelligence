# Hybrid retrieval with PostgreSQL and pgvector

## Design

A dedicated `PgVectorStore` implements the repository boundary. An embedding adapter generates 384-dimensional vectors before persistence; a provider is not hard-coded into the store. The schema stores tenant, document, chunk, source and embedding metadata. PostgreSQL generated `tsvector` supports lexical retrieval; pgvector cosine distance supports semantic retrieval.

Search combines the top lexical and semantic candidates with reciprocal rank fusion (RRF, k=60), preserving source and chunk identifiers for citations.

## Tenant isolation

Tenant IDs must be obtained from verified identity claims, never arbitrary user headers. Every repository operation establishes transaction-local `app.tenant_id`. The database uses `FORCE ROW LEVEL SECURITY` with both `USING` and `WITH CHECK`. Runtime roles must not own tables or have `BYPASSRLS`.

## Deployment notes

- Requires PostgreSQL with the `vector` extension; plain postgres:17 images generally do not include it.
- The embedding dimension is fixed at 384. Changing providers requires versioned embeddings or a migration/reindex strategy.
- The sample uses English full-text stemming; add per-language configuration for Portuguese documents.
- HNSW and FTS indexes have memory/write costs; benchmark recall and latency with tenant-filtered workloads.
- Do not claim model-generated answers, GraphRAG or production-scale benchmarks: these are separate milestones.

## Next steps

Implement an embedding provider adapter, document metadata and idempotent upserts, end-to-end RLS tests using a non-owner role, and an optional grounded answer-generation layer. Keep graph entities/edges in a separate bounded subsystem until hybrid retrieval has measured quality.
