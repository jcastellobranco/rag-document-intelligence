FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir '.[pdf]' && useradd -u 10001 -m app
USER app
EXPOSE 8000
CMD ["uvicorn","rag_intelligence.service:create_app","--factory","--host","0.0.0.0","--port","8000"]
