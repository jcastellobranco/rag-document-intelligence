-- Local test role only; use a secret manager and separate migrations in deployment.
DO $$
BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='rag_runtime') THEN
  CREATE ROLE rag_runtime LOGIN PASSWORD 'local-rag-development-only';
 END IF;
END $$;
ALTER ROLE rag_runtime NOBYPASSRLS;
GRANT CONNECT ON DATABASE rag TO rag_runtime;
GRANT USAGE ON SCHEMA public TO rag_runtime;
GRANT SELECT, INSERT, DELETE ON rag_chunks TO rag_runtime;
