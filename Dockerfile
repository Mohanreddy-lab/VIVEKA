FROM python:3.11-slim

WORKDIR /app

# System deps for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ollama runs on the host; container connects via OLLAMA_HOST.
# Docker Desktop host bridge: http://host.docker.internal:11434
# Or set LLM_PROVIDER=huggingface to use HF Inference API (no Ollama needed).
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV LLM_PROVIDER=ollama
ENV VIVEKA_MODEL=llama3.2

VOLUME ["/app/data"]

EXPOSE 7860
CMD ["python", "app.py"]
