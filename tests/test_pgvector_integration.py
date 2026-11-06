"""Run against migrated pgvector PostgreSQL with non-owner RLS runtime role."""
import os
from uuid import uuid4
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from rag_intelligence.vector_store import PgVectorStore

@pytest.mark.integration
def test_hybrid_search_and_cross_tenant_isolation():
    url = os.getenv("RAG_TEST_DATABASE_URL")
    if not url:
        pytest.skip("RAG_TEST_DATABASE_URL is not configured")
    engine = create_engine(url)
    sessions = sessionmaker(bind=engine)
    tenant_a, tenant_b, document = uuid4(), uuid4(), uuid4()
    chunk = uuid4()
    embedding = [0.0] * 384
    embedding[0] = 1.0
    store = PgVectorStore(sessions)
    with engine.connect() as db:
        role, owner, bypass = db.execute(text(
            "SELECT current_user, pg_get_userbyid(c.relowner), r.rolbypassrls "
            "FROM pg_class c JOIN pg_roles r ON r.rolname=current_user "
            "WHERE c.oid='rag_chunks'::regclass"
        )).one()
        assert role != owner and not bypass
    try:
        store.add_chunks(tenant_a, document, [{
            "id":chunk,"source":"manual.md","ordinal":0,
            "content":"Kubernetes schedules pods onto nodes",
            "embedding":embedding
        }])
        hits = store.search(tenant_a, "Kubernetes pods", embedding)
        assert hits and hits[0].chunk_id == str(chunk)
        assert store.search(tenant_b, "Kubernetes pods", embedding) == []
    finally:
        with sessions.begin() as db:
            store._tenant(db, tenant_a)
            db.execute(text("DELETE FROM rag_chunks WHERE id=:id"), {"id":str(chunk)})
    engine.dispose()

@pytest.mark.integration
def test_embedding_pipeline_against_postgres_with_tenant_guard():
    from rag_intelligence.pipeline import HybridRagPipeline
    from rag_intelligence.embeddings import SentenceTransformerEmbedder

    class DeterministicModel:
        def encode(self, texts, normalize_embeddings=True):
            return [[1.0] + [0.0] * 383 for _ in texts]

    url = os.environ["RAG_TEST_DATABASE_URL"]
    engine = create_engine(url)
    tenant_a, tenant_b = uuid4(), uuid4()
    store = PgVectorStore(sessionmaker(bind=engine))
    pipeline = HybridRagPipeline(store, SentenceTransformerEmbedder(model=DeterministicModel()))
    document = None
    try:
        document = pipeline.ingest(tenant_a, "pipeline.md", "PostgreSQL supports row level security")
        hits = pipeline.search(tenant_a, "row level security")
        assert any(hit.source == "pipeline.md" for hit in hits)
        assert pipeline.search(tenant_b, "row level security") == []
    finally:
        if document:
            with sessionmaker(bind=engine).begin() as db:
                store._tenant(db, tenant_a)
                db.execute(text("DELETE FROM rag_chunks WHERE document_id=:id"),
                           {"id": document["document_id"]})
        engine.dispose()
