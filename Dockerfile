FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data /app/data
COPY . .

ENV PYTHONPATH="/app:/app/data_handle:/app/features:/app/models:/app/plots:/app/analysis"

CMD ["sleep", "infinity"]