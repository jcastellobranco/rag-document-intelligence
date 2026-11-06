import math
import pytest
from rag_intelligence.vector_store import _vector, DIMENSIONS, PgVectorStore

def test_vector_dimensions_and_nonfinite_values():
    assert _vector([0.0] * DIMENSIONS).startswith("[")
    with pytest.raises(ValueError):
        _vector([0.1] * (DIMENSIONS - 1))
    with pytest.raises(ValueError):
        _vector([math.nan] * DIMENSIONS)

def test_retrieval_limit_validation():
    store = PgVectorStore(None)
    from uuid import uuid4
    with pytest.raises(ValueError):
        store.search(uuid4(), "question", [0.0] * DIMENSIONS, limit=0)
    assert store.search(uuid4(), " ", [0.0] * DIMENSIONS) == []
