FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY ""
ENV PINECONE_API_KEY ""
ENV PINECONE_ENV ""
ENV DATASET_PATH "dataset/indonesia-ai-dataset.pdf"
ENV INDEX_NAME "iai-chatbot"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]