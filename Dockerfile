FROM python:3.9-slim

WORKDIR /app

# Εγκατάσταση system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements πρώτα για better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy όλο τον κώδικα
COPY . .

# Ορισμός Python path για όλα τα modules
ENV PYTHONPATH="/app:/app/data_handle:/app/features:/app/models:/app/plots:/app/analysis"

# Κρατάει το container ανοιχτό για VSCode
CMD ["sleep", "infinity"]