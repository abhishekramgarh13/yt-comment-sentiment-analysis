
# Stage 1: Builder Stage
FROM python:3.10 as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application files and requirements
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl
COPY requirements.txt /app/requirements.txt

# Install dependencies into a temporary directory
RUN pip install --no-cache-dir --target=/app/dependencies -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

---

# Stage 2: Final Image
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependencies from the builder stage
COPY --from=builder /app/dependencies /usr/local/lib/python3.10/site-packages/

# Copy application files
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

# Expose the application port
EXPOSE 5000

# Run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
