FROM python:3.11-slim

WORKDIR /app

# System deps for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ollama runs on the host; container connects via OLLAMA_HOST env var.
# Default points to Docker Desktop host bridge.
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV LLM_PROVIDER=ollama
ENV VIVEKA_MODEL=llama3.2

# sentence-transformers downloads the model on first use.
# Pre-warm it at build time so the first request isn't slow.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

VOLUME ["/app/data"]

# Default: API server. Override to run the Streamlit UI.
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
