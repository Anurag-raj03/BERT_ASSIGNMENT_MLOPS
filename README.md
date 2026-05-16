# Clinical BERT Real-Time Inference API

This is a real-time inference API for clinical text classification using Hugging Face's `bvanaken/clinical-assertion-negation-bert`.

## Features
- **FastAPI** backend exposing `/predict` and `/health`
- **Instant Boot:** Docker image pre-fetches the Hugging Face model so the container boots in milliseconds
- **CI/CD:** GitHub Actions to automatically run `pytest` and deploy to Google Cloud Run

## How to Run Locally

### 1. Using Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

### 2. Using Docker
```bash
docker build -t clinical-bert-api .
docker run -p 8080:8080 clinical-bert-api
```

## Testing the API
Once the server is running, you can test it natively by going to:
[http://localhost:8080/docs](http://localhost:8080/docs) (Swagger UI)

Or via `curl`:
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"sentence": "The patient denies chest pain."}'
```
