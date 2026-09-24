FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for sentence-transformers and PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create necessary directories
RUN mkdir -p data/documents data/golden data/chroma evidence

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
