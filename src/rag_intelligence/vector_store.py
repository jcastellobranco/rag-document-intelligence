"""PostgreSQL/pgvector retrieval repository; tenant context is transaction-local.

Embeddings are supplied by an adapter; this module does not call an LLM.
"""
from dataclasses import dataclass
from typing import Sequence
from uuid import UUID
from sqlalchemy import text
from sqlalchemy.orm import Session

DIMENSIONS = 384

@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    source: str
    content: str
    score: float
    method: str

def _vector(values: Sequence[float]) -> str:
    import math
    if len(values) != DIMENSIONS or not all(math.isfinite(float(x)) for x in values):
        raise ValueError(f"Expected {DIMENSIONS} finite embedding dimensions")
    return "[" + ",".join(format(float(x), ".9g") for x in values) + "]"

class PgVectorStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def add_chunks(self, tenant_id: UUID, document_id: UUID, chunks: Sequence[dict]) -> int:
        """Insert document chunks in one transaction; RLS is enforced by PostgreSQL."""
        with self.session_factory.begin() as session:
            self._tenant(session, tenant_id)
            for chunk in chunks:
                session.execute(text("""
                    INSERT INTO rag_chunks(id, tenant_id, document_id, source, ordinal, content, embedding)
                    VALUES (:id, :tenant, :document, :source, :ordinal, :content, CAST(:embedding AS vector))
                """), {"id": str(UUID(str(chunk["id"]))), "tenant": str(tenant_id),
                       "document": str(document_id), "source": chunk["source"],
                       "ordinal": chunk["ordinal"], "content": chunk["content"],
                       "embedding": _vector(chunk["embedding"])})
        return len(chunks)

    @staticmethod
    def _tenant(session: Session, tenant_id: UUID) -> None:
        session.execute(text("SELECT set_config('app.tenant_id', :tenant, true)"),
                        {"tenant": str(tenant_id)})

    def search(self, tenant_id: UUID, question: str, embedding: Sequence[float], limit: int = 5) -> list[RetrievalHit]:
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        if not question.strip():
            return []
        vector = _vector(embedding)
        with self.session_factory.begin() as session:
            self._tenant(session, tenant_id)
            # Rank lexical and semantic candidates independently and fuse using RRF.
            rows = session.execute(text("""
                WITH lexical AS (
                    SELECT id, row_number() OVER (
                        ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', :query)) DESC, id
                    ) AS rank
                    FROM rag_chunks
                    WHERE tenant_id=:tenant AND search_vector @@ websearch_to_tsquery('english', :query)
                    ORDER BY rank LIMIT 50
                ), semantic AS (
                    SELECT id, row_number() OVER (
                        ORDER BY embedding <=> CAST(:embedding AS vector), id
                    ) AS rank
                    FROM rag_chunks WHERE tenant_id=:tenant
                    ORDER BY rank LIMIT 50
                ), fused AS (
                    SELECT id, sum(score) AS score FROM (
                        SELECT id, 1.0/(60+rank) AS score FROM lexical
                        UNION ALL
                        SELECT id, 1.0/(60+rank) AS score FROM semantic
                    ) ranked GROUP BY id
                )
                SELECT c.id, c.source, c.content, f.score
                FROM fused f JOIN rag_chunks c ON c.id=f.id
                WHERE c.tenant_id=:tenant
                ORDER BY f.score DESC, c.id LIMIT :limit
            """), {"tenant": str(tenant_id), "query": question,
                   "embedding": vector, "limit": limit}).mappings().all()
        return [RetrievalHit(str(r["id"]), r["source"], r["content"],
                             float(r["score"]), "hybrid_rrf") for r in rows]

    def list_documents(self, tenant_id: UUID, limit: int = 100) -> list[dict]:
        """Persistent tenant-scoped catalogue; no document content is returned."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self.session_factory.begin() as session:
            self._tenant(session, tenant_id)
            rows = session.execute(text("""
                SELECT document_id, min(source) AS source, count(*) AS chunks,
                       min(created_at) AS created_at
                FROM rag_chunks
                WHERE tenant_id = :tenant
                GROUP BY document_id
                ORDER BY created_at DESC, document_id DESC
                LIMIT :limit
            """), {"tenant": str(tenant_id), "limit": limit}).mappings().all()
            return [
                {"document_id": str(row["document_id"]), "source": row["source"],
                 "chunks": row["chunks"], "created_at": row["created_at"].isoformat()}
                for row in rows
            ]

    def get_document(self, tenant_id: UUID, document_id: UUID) -> dict | None:
        """Read only the selected document under tenant RLS; never fall back to corpus search.

        Ingestion is limited to 200,000 characters, producing at most 471 chunks
        with the standard chunker. Fail closed if this limit is exceeded.
        """
        with self.session_factory.begin() as session:
            self._tenant(session, tenant_id)
            rows = session.execute(text("""
                SELECT id, source, ordinal, content
                FROM rag_chunks
                WHERE tenant_id = :tenant AND document_id = :document
                ORDER BY ordinal
                LIMIT 513
            """), {"tenant": str(tenant_id), "document": str(document_id)}).mappings().all()
        if not rows:
            return None
        if len(rows) > 512:
            raise ValueError("Selected document exceeds the audit chunk limit")
        return {
            "document_id": str(document_id), "source": rows[0]["source"],
            "chunks": [{"chunk_id": str(row["id"]), "ordinal": row["ordinal"],
                        "text": row["content"]} for row in rows],
        }

    def delete_document(self, tenant_id: UUID, document_id: UUID) -> bool:
        """Delete only chunks visible through tenant RLS; returns False if not found."""
        with self.session_factory.begin() as session:
            self._tenant(session, tenant_id)
            deleted = session.execute(text("""
                DELETE FROM rag_chunks
                WHERE tenant_id = :tenant AND document_id = :document
            """), {"tenant": str(tenant_id), "document": str(document_id)})
            return deleted.rowcount > 0
