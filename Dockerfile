FROM python:3.10 AS builder

WORKDIR /app

COPY flask_app/ /app/

COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN pip install -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install gunicorn

RUN python -m nltk.downloader stopwords wordnet

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY --from=builder /app /app

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]



