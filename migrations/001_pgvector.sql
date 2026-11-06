-- Execute as migration owner, never as the tenant-facing application role.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE TABLE IF NOT EXISTS rag_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL,
    document_id uuid NOT NULL,
    source text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    content text NOT NULL CHECK (length(content) BETWEEN 1 AND 10000),
    embedding vector(384) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS rag_chunks_tenant_idx ON rag_chunks(tenant_id);
CREATE INDEX IF NOT EXISTS rag_chunks_fts_idx ON rag_chunks USING gin(search_vector);
CREATE INDEX IF NOT EXISTS rag_chunks_vector_idx ON rag_chunks
    USING hnsw (embedding vector_cosine_ops);
ALTER TABLE rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_chunks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rag_chunks_tenant_policy ON rag_chunks;
CREATE POLICY rag_chunks_tenant_policy ON rag_chunks
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid);
-- Grant SELECT/INSERT/DELETE to a separate non-owner runtime role.
-- This migration intentionally does not create database users or embed passwords.
