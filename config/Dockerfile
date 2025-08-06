# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Copy only dependency specs first for better caching
COPY pyproject.toml poetry.lock* /app/

# Install build tools and our package in editable mode
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends curl build-essential \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir . \
    && apt-get purge -y --auto-remove curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . /app

# Expose any ports if you later add a web server
# EXPOSE 8000

# Default entrypoint: launch the REPL
ENTRYPOINT ["meta-model"]

