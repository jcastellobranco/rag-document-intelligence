# rag-document-intelligence

Local document indexing and transparent citation retrieval using deterministic lexical scoring.

## Status

Early-stage educational prototype. Not production-ready. No proprietary course materials, credentials or private datasets are included.

## Current implementation

The Python domain logic is available under `src/`. This first commit establishes a minimal testable baseline; packaging, API integration, automated tests, Docker, CI and operational hardening are being added incrementally.

## Security

Do not deploy the demonstration components as a multi-tenant public service. Authentication, durable persistence, rate limiting and observability require additional work.

## Roadmap

- Complete package metadata and automated tests
- Add an executable example and continuous integration
- Implement production-oriented providers, tracing and deployment guidance
