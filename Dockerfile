FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY code_indexer_server.py .

# Create directory for ChromaDB
RUN mkdir -p /app/chroma_db

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "code_indexer_server.py"] 