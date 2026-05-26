FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — CPU-only torch to keep image lean
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    sentence-transformers>=3.0 \
    chromadb>=0.5 \
    google-genai>=0.8 \
    flask>=3.0 \
    flask-cors>=4.0 \
    python-dotenv>=1.0

# Copy source and pre-download the embedding model into its own layer.
# This layer is cached — subsequent rebuilds skip the ~1.2 GB download
# as long as src/ and the pip layers above are unchanged.
COPY src/ ./src/
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

# Copy pre-chunked data (committed to repo) and build ChromaDB indices.
# indices/ is gitignored — we build it here instead of copying it.
COPY data/processed/ ./data/processed/
COPY build_index.py .
RUN python build_index.py

# Copy remaining static assets
COPY docs/ ./docs/
COPY server.py .

EXPOSE 8080

CMD ["python", "-c", "import server; server.app.run(host='0.0.0.0', port=8080, debug=False)"]
