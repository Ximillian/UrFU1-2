FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./artifacts_books_reco ./artifacts_books_reco/
COPY ./server.py .

EXPOSE 8080

CMD ["python", "server.py"]