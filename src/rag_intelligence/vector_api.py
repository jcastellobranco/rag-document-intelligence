"""Tenant-bound HTTP API backed by PostgreSQL pgvector hybrid retrieval.

The service has a *single configured tenant* and authenticates a service API
key. Multi-tenant callers must route only their authorized tenant here.
"""
import hmac
import os
from uuid import UUID
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .embeddings import SentenceTransformerEmbedder
from .pipeline import HybridRagPipeline
from .vector_store import PgVectorStore

class DocumentIn(BaseModel):
    source: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=200000)

class SearchIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)

def create_app(*, api_key: str | None = None, tenant_id: str | None = None,
               pipeline: HybridRagPipeline | None = None) -> FastAPI:
    secret = api_key if api_key is not None else os.getenv("RAG_API_KEY", "")
    raw_tenant = tenant_id if tenant_id is not None else os.getenv("RAG_TENANT_ID", "")
    if not secret:
        raise RuntimeError("RAG_API_KEY is required")
    try:
        tenant = UUID(raw_tenant)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("RAG_TENANT_ID must be a valid UUID") from exc
    if pipeline is None:
        database_url = os.getenv("RAG_DATABASE_URL")
        if not database_url:
            raise RuntimeError("RAG_DATABASE_URL is required for vector API")
        engine = create_engine(database_url, pool_pre_ping=True, pool_size=5)
        pipeline = HybridRagPipeline(
            PgVectorStore(sessionmaker(bind=engine)),
            SentenceTransformerEmbedder(model_name=os.getenv(
                "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )),
        )
    app = FastAPI(title="Vector Document Intelligence", version="0.4.0")

    def identity(x_api_key: str | None = Header(None),
                 x_tenant_id: str | None = Header(None)) -> UUID:
        if not x_api_key or not hmac.compare_digest(secret, x_api_key):
            raise HTTPException(401, "Invalid API key")
        if x_tenant_id is not None and x_tenant_id != str(tenant):
            raise HTTPException(403, "Tenant mismatch")
        return tenant

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "pgvector"}

    @app.post("/v1/documents")
    def ingest(body: DocumentIn, authorized_tenant: UUID = Depends(identity)):
        return pipeline.ingest(authorized_tenant, body.source, body.text)

    @app.post("/v1/search")
    def search(body: SearchIn, authorized_tenant: UUID = Depends(identity)):
        hits = pipeline.search(authorized_tenant, body.question, body.limit)
        return {"citations": [
            {"chunk_id": h.chunk_id, "source": h.source, "text": h.content,
             "score": h.score, "method": h.method}
            for h in hits
        ]}
    return app
