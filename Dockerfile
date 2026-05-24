FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — CPU-only torch to keep image lean
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    sentence-transformers>=3.0 \
    chromadb>=0.5 \
    google-genai>=0.8 \
    flask>=3.0 \
    flask-cors>=4.0 \
    python-dotenv>=1.0

# Copy source code and assets
COPY src/ ./src/
COPY docs/ ./docs/
COPY indices/ ./indices/
COPY server.py .

# Pre-download the embedding model into the image layer (avoids cold-start download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

EXPOSE 8080

CMD ["python", "-c", "import server; server.app.run(host='0.0.0.0', port=8080, debug=False)"]
