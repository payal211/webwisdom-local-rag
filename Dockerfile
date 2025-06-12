# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/.cache \
    TRANSFORMERS_CACHE=/app/models/.cache \
    TORCH_HOME=/app/models/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=120 -r requirements.txt && \
    python -c "import torch; print('PyTorch version:', torch.__version__)" && \
    python -c "import transformers; print('Transformers version:', transformers.__version__)" && \
    python -c "import sentence_transformers; print('Sentence-transformers version:', sentence_transformers.__version__)" && \
    python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"

# Download NLTK data (if using NLTK)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null || echo "NLTK download skipped"

# Copy application code
COPY . .

# Create directories for data and models with proper permissions
RUN mkdir -p /app/data /app/models /app/models/.cache && \
    chmod -R 755 /app/data /app/models

# Pre-download a small model to verify setup (optional)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models/.cache')" 2>/dev/null || echo "Model pre-download skipped"

# Expose ports
EXPOSE 8501 5000

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - can be overridden by docker-compose
CMD ["python", "-m", "streamlit", "run", "st_app.py", "--server.port=8501", "--server.address=0.0.0.0"]