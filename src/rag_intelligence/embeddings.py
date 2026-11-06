"""Embedding provider boundary with an optional local sentence-transformers model."""
from collections.abc import Sequence
from math import isfinite, sqrt
from .vector_store import DIMENSIONS

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def validate_embeddings(vectors: Sequence[Sequence[float]], expected: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise ValueError("Embedding provider returned an unexpected number of vectors")
    result = []
    for vector in vectors:
        if len(vector) != DIMENSIONS:
            raise ValueError(f"Expected {DIMENSIONS} dimensions")
        values = [float(value) for value in vector]
        if not all(isfinite(value) for value in values):
            raise ValueError("Embeddings must be finite")
        norm = sqrt(sum(value * value for value in values))
        if norm == 0:
            raise ValueError("Zero embeddings are not supported for cosine search")
        result.append([value / norm for value in values])
    return result

class SentenceTransformerEmbedder:
    """Lazy model load: installation is optional and tests can inject a fake model."""
    def __init__(self, model_name: str = DEFAULT_MODEL, model=None):
        self.model_name = model_name
        self._model = model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text must be a nonempty string")
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install the 'embeddings' extra to use the local model") from exc
            self._model = SentenceTransformer(self.model_name)
        raw = self._model.encode(list(texts), normalize_embeddings=True)
        return validate_embeddings(raw, len(texts))
