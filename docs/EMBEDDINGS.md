# Local embeddings adapter

Install the optional model runtime with `pip install -e '.[vector,embeddings]'`. The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). The first use downloads model weights unless they have already been cached. Do not put model downloads in the normal CI test path.

`SentenceTransformerEmbedder` loads lazily, validates shape and finite values, and normalizes vectors for cosine search. `HybridRagPipeline` connects it to the existing tenant-scoped PostgreSQL repository for chunk ingestion and hybrid retrieval.

Unit tests inject a deterministic fake model to validate the adapter without network or GPU access. **This is not yet an end-to-end test using downloaded model weights.** Run a separate model smoke test in an environment with the weights available before claiming that capability.

For multilingual or Portuguese corpora, benchmark a multilingual embedding model with a compatible 384-dimensional output (or migrate the schema), and measure retrieval recall before choosing it. Changing models without re-embedding existing documents makes similarity rankings invalid.
