FROM python:3.9

# Use NVIDIA PyTorch container as base
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install a specific version of sqlparse first
RUN pip install mlflow==2.19.0

# Copy project
COPY . .