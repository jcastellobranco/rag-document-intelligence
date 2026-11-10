"""Tenant-bound HTTP API backed by PostgreSQL pgvector hybrid retrieval.

The service has a *single configured tenant* and authenticates a service API
key. Multi-tenant callers must route only their authorized tenant here.
"""
import hmac
import io
import logging
import os
import time
from uuid import UUID
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .embeddings import SentenceTransformerEmbedder
from .pipeline import HybridRagPipeline
from .vector_store import PgVectorStore
from .metrics import (metrics_app, operation_for, HTTP_REQUESTS, HTTP_DURATION,
                      SEARCH_DURATION, SEARCH_HITS, INGEST_CHUNKS)

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
    app.mount("/metrics", metrics_app)
    logger = logging.getLogger("uvicorn.error")

    @app.middleware("http")
    async def record_http_metrics(request: Request, call_next):
        if request.url.path.startswith("/metrics") or request.url.path == "/health":
            return await call_next(request)
        operation = operation_for(request.method, request.url.path)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            HTTP_REQUESTS.labels(operation=operation, status_class=f"{status // 100}xx").inc()
            duration = time.perf_counter() - started
            HTTP_DURATION.labels(operation=operation).observe(duration)
            logger.info("rag_http operation=%s status_class=%s duration_ms=%.1f",
                        operation, f"{status // 100}xx", duration * 1000)

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
        result = pipeline.ingest(authorized_tenant, body.source, body.text)
        if isinstance(result, dict) and isinstance(result.get("chunks"), int):
            INGEST_CHUNKS.labels(format="text").inc(result["chunks"])
        return result

    @app.get("/v1/documents")
    def list_documents(authorized_tenant: UUID = Depends(identity)):
        return {"items": pipeline.list_documents(authorized_tenant)}

    @app.get("/v1/documents/{document_id}/content")
    def document_content(document_id: UUID, authorized_tenant: UUID = Depends(identity)):
        """Private authenticated document-scoped read for the audit worker."""
        try:
            document = pipeline.get_document(authorized_tenant, document_id)
        except ValueError:
            raise HTTPException(413, "Selected document is too large to audit")
        if document is None:
            raise HTTPException(404, "Document not found")
        return document

    @app.delete("/v1/documents/{document_id}")
    def delete_document(document_id: UUID, authorized_tenant: UUID = Depends(identity)):
        if not pipeline.delete_document(authorized_tenant, document_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": True}

    @app.post("/v1/documents/pdf")
    async def ingest_pdf(file: UploadFile = File(...), authorized_tenant: UUID = Depends(identity)):
        if file.content_type != "application/pdf":
            raise HTTPException(415, "Only application/pdf is supported")
        raw = await file.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "PDF exceeds 10 MiB")
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw), strict=True)
            if len(reader.pages) > 100:
                raise HTTPException(413, "PDF exceeds 100 pages")
            parts = []
            total_chars = 0
            for page in reader.pages:
                page_text = page.extract_text() or ""
                total_chars += len(page_text) + (1 if parts else 0)
                if total_chars > 200000:
                    raise HTTPException(413, "Extracted text exceeds 200000 characters")
                parts.append(page_text)
            extracted = "\n".join(parts)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, "Could not parse PDF") from exc
        if not extracted.strip():
            raise HTTPException(422, "No extractable text in PDF")
        if len(extracted) > 200000:
            raise HTTPException(413, "Extracted text exceeds 200000 characters")
        result = pipeline.ingest(authorized_tenant, (file.filename or "upload.pdf")[:256], extracted)
        if isinstance(result, dict) and isinstance(result.get("chunks"), int):
            INGEST_CHUNKS.labels(format="pdf").inc(result["chunks"])
        return result

    @app.post("/v1/search")
    def search(body: SearchIn, authorized_tenant: UUID = Depends(identity)):
        started = time.perf_counter()
        try:
            hits = pipeline.search(authorized_tenant, body.question, body.limit)
        finally:
            SEARCH_DURATION.observe(time.perf_counter() - started)
        SEARCH_HITS.observe(len(hits))
        return {"citations": [
            {"chunk_id": h.chunk_id, "source": h.source, "text": h.content,
             "score": h.score, "method": h.method}
            for h in hits
        ]}
    return app
