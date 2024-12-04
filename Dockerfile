FROM python:3.10 as build

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y libgomp1

# Copy application files and requirements
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl


# Install dependencies into a temporary directory
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet


FROM python:3.10-slim as final

WORKDIR /app


COPY --from=build /app /app


EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]



