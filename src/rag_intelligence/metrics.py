"""Bounded-label RAG metrics. Never use text, source, tenant or document IDs as labels."""
from prometheus_client import Counter, Histogram, make_asgi_app

metrics_app = make_asgi_app()

HTTP_REQUESTS = Counter(
    "atlas_rag_http_requests_total", "RAG HTTP requests by operation and status class",
    ["operation", "status_class"],
)
HTTP_DURATION = Histogram(
    "atlas_rag_http_request_duration_seconds", "RAG HTTP request duration",
    ["operation"],
)
SEARCH_DURATION = Histogram(
    "atlas_rag_search_duration_seconds", "RAG hybrid retrieval duration",
)
SEARCH_HITS = Histogram(
    "atlas_rag_search_hits", "Number of chunks returned per successful search",
    buckets=(0, 1, 2, 3, 5, 10, 20),
)
INGEST_CHUNKS = Counter(
    "atlas_rag_ingested_chunks_total", "Successfully ingested document chunks",
    ["format"],
)
EMBEDDING_DURATION = Histogram(
    "atlas_rag_embedding_duration_seconds", "Embedding model encode duration",
)
EMBEDDING_TEXTS = Counter(
    "atlas_rag_embedding_texts_total", "Texts successfully embedded",
)


def operation_for(method: str, path: str) -> str:
    """Classify fixed routes, never export raw paths or IDs."""
    if method == "POST" and path == "/v1/search":
        return "search"
    if method == "POST" and path == "/v1/documents/pdf":
        return "ingest_pdf"
    if method == "POST" and path == "/v1/documents":
        return "ingest_text"
    if method == "GET" and path == "/v1/documents":
        return "list_documents"
    if method == "GET" and path.startswith("/v1/documents/") and path.endswith("/content"):
        return "read_document"
    if method == "DELETE" and path.startswith("/v1/documents/"):
        return "delete_document"
    return "other"
