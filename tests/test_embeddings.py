import math
import pytest
from uuid import uuid4
from rag_intelligence.embeddings import SentenceTransformerEmbedder, validate_embeddings
from rag_intelligence.pipeline import HybridRagPipeline

class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        assert normalize_embeddings is True
        return [[1.0] + [0.0] * 383 for _ in texts]

class FakeStore:
    def __init__(self):
        self.saved = []
    def add_chunks(self, tenant, document, chunks):
        self.saved.extend(chunks)
        return len(chunks)
    def search(self, tenant, question, embedding, limit):
        assert len(embedding) == 384
        return ["hit"]

def test_real_model_adapter_with_fake_model():
    assert SentenceTransformerEmbedder(model=FakeModel()).encode(["hello"])[0][0] == 1.0

def test_validation_rejects_bad_vectors():
    with pytest.raises(ValueError, match="number"):
        validate_embeddings([], 1)
    with pytest.raises(ValueError, match="dimensions"):
        validate_embeddings([[1.0]], 1)
    with pytest.raises(ValueError, match="finite"):
        validate_embeddings([[math.nan] + [0.0] * 383], 1)
    with pytest.raises(ValueError, match="Zero"):
        validate_embeddings([[0.0] * 384], 1)

def test_pipeline_ingestion_and_search():
    store = FakeStore()
    pipeline = HybridRagPipeline(store, SentenceTransformerEmbedder(model=FakeModel()))
    result = pipeline.ingest(uuid4(), "manual.md", "Kubernetes schedules pods")
    assert result["chunks"] == 1
    assert len(store.saved) == 1
    assert pipeline.search(uuid4(), "pods") == ["hit"]
