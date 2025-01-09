FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies (for PyPDF2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dataset. VERY IMPORTANT!
COPY dataset /app/dataset

COPY . .

# Set environment variables within the Dockerfile (for development/testing ONLY)
# For production, use docker-compose or environment variables when running the container
ENV OPENAI_API_KEY=your_openai_api_key
ENV PINECONE_API_KEY=your_pinecone_api_key
ENV PINECONE_ENV=your_pinecone_environment

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]