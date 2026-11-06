"""Ingest and query pgvector with one injectable embedding provider."""
from uuid import UUID, uuid4
from .advanced import chunk_text
from .embeddings import validate_embeddings

class HybridRagPipeline:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def ingest(self, tenant_id: UUID, source: str, text: str) -> dict:
        if not source.strip() or not text.strip():
            raise ValueError("source and text must not be blank")
        chunks = chunk_text(source, text)
        embeddings = validate_embeddings(self.embedder.encode([c.text for c in chunks]), len(chunks))
        document_id = uuid4()
        records = [
            {"id": uuid4(), "source": c.source, "ordinal": c.ordinal,
             "content": c.text, "embedding": vector}
            for c, vector in zip(chunks, embeddings)
        ]
        inserted = self.store.add_chunks(tenant_id, document_id, records)
        return {"document_id": str(document_id), "chunks": inserted}

    def search(self, tenant_id: UUID, question: str, limit: int = 5):
        if not question.strip():
            return []
        embeddings = validate_embeddings(self.embedder.encode([question]), 1)
        return self.store.search(tenant_id, question, embeddings[0], limit)

    def list_documents(self, tenant_id: UUID, limit: int = 100) -> list[dict]:
        return self.store.list_documents(tenant_id, limit)

    def delete_document(self, tenant_id: UUID, document_id: UUID) -> bool:
        return self.store.delete_document(tenant_id, document_id)
