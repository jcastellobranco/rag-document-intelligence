"""Persistent, authenticated, single-tenant document retrieval using SQLite FTS5."""
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager
from threading import RLock
from uuid import uuid4
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

class DocumentIn(BaseModel):
    source: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=200000)

class SearchIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)

class Store:
    def __init__(self, path: str):
        self.path = path
        self.lock = RLock()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, tenant TEXT NOT NULL, source TEXT NOT NULL, ordinal INTEGER NOT NULL, body TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS chunks_tenant ON chunks(tenant)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, content='chunks', content_rowid='rowid')")
            db.executescript("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid,body) VALUES(new.rowid,new.body);
                END;
            """)
    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()
    def add(self, tenant: str, doc: DocumentIn) -> dict:
        doc_id = str(uuid4())
        chunks = []
        size, overlap = 500, 75
        for ordinal, start in enumerate(range(0, len(doc.text), size-overlap)):
            chunk = doc.text[start:start+size]
            if chunk:
                chunks.append((f"{doc_id}:{ordinal}", tenant, doc.source, ordinal, chunk))
            if start + size >= len(doc.text):
                break
        with self.lock, self.connect() as db:
            db.executemany("INSERT INTO chunks(id,tenant,source,ordinal,body) VALUES(?,?,?,?,?)", chunks)
        return {"document_id": doc_id, "chunks": len(chunks)}
    def search(self, tenant: str, question: str, limit: int) -> list[dict]:
        terms = re.findall(r"\w+", question)
        if not terms:
            return []
        match = " OR ".join('"' + word + '"' for word in terms[:20])
        with self.lock, self.connect() as db:
            rows = db.execute("""
                SELECT c.id,c.source,c.ordinal,c.body,bm25(chunks_fts) AS rank
                FROM chunks_fts JOIN chunks c ON c.rowid=chunks_fts.rowid
                WHERE chunks_fts MATCH ? AND c.tenant=? ORDER BY rank LIMIT ?
            """, (match, tenant, limit)).fetchall()
        return [{"chunk_id":r["id"],"source":r["source"],"ordinal":r["ordinal"],"text":r["body"],"score":round(-r["rank"],6)} for r in rows]

def create_app(path: str | None = None, api_key: str | None = None) -> FastAPI:
    key = api_key if api_key is not None else os.getenv("RAG_API_KEY","")
    if not key:
        raise RuntimeError("RAG_API_KEY required; unauthenticated startup is disabled")
    store = Store(path or os.getenv("RAG_DB_PATH","rag.sqlite3"))
    app = FastAPI(title="Document Intelligence",version="0.2.0")
    def identity(x_api_key: str | None = Header(default=None), x_tenant_id: str | None = Header(default=None)) -> str:
        if not x_api_key or not hmac.compare_digest(x_api_key,key):
            raise HTTPException(401,"Invalid API key")
        tenant = os.getenv("RAG_TENANT_ID","demo")
        if x_tenant_id is not None and x_tenant_id != tenant:
            raise HTTPException(403,"Tenant mismatch")
        return tenant
    @app.get("/health")
    def health():
        return {"status":"ok"}
    @app.post("/v1/documents")
    def ingest(doc: DocumentIn, tenant: str = Depends(identity)):
        return store.add(tenant,doc)
    @app.post("/v1/search")
    def search(query: SearchIn, tenant: str = Depends(identity)):
        return {"citations":store.search(tenant,query.question,query.limit)}
    return app
