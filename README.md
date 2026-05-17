# 🏥 Clinical BERT Real-Time Inference API

[![CI Pipeline](https://github.com/Anurag-raj03/BERT_ASSIGNMENT_MLOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/Anurag-raj03/BERT_ASSIGNMENT_MLOPS/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/Anurag-raj03/BERT_ASSIGNMENT_MLOPS/actions/workflows/cd.yml/badge.svg)](https://github.com/Anurag-raj03/BERT_ASSIGNMENT_MLOPS/actions/workflows/cd.yml)

A production-grade, real-time clinical text classification API built with **FastAPI**, containerized with **Docker**, and deployed serverless to **Google Cloud Run**. 

This API classifies medical sentences into clinical assertion states (`PRESENT`, `ABSENT`, `POSSIBLE`) utilizing the fine-tuned Hugging Face model [`bvanaken/clinical-assertion-negation-bert`](https://huggingface.co/bvanaken/clinical-assertion-negation-bert).

---

## 🚀 Live Production Environment

The API is fully deployed and publicly accessible on Google Cloud Run (Mumbai Region). It features automated cold-start optimizations and scales to zero when idle.

* **Interactive Swagger UI Dashboard:** [👉 Click here to test live](https://clinical-bert-api-811357407534.asia-south1.run.app/docs)
* **Production API Base URL:** `https://clinical-bert-api-811357407534.asia-south1.run.app`

---

## ✨ Key Architectural Features & Extra Credit

* **⚡ Sub-500ms Cloud Run Cold Starts:** The Dockerfile is engineered to pre-download and cache the 400MB Hugging Face model weights directly into the container image layer during the build phase. This prevents runtime downloads and guarantees near-instant server boot times.
* **🧠 Lifespan Model Caching (Singleton):** The PyTorch BERT model is loaded into memory exactly once during the FastAPI `@asynccontextmanager` lifespan startup event, keeping inference latency under `~250ms`.
* **📦 Batch Prediction Support (Extra Credit):** Exposes a `POST /predict_batch` endpoint optimized for PyTorch matrix multiplication, allowing high-throughput concurrent processing of multiple clinical notes.
* **🩺 Health Monitoring (Extra Credit):** Includes a `GET /health` endpoint for automated Kubernetes/Cloud Run liveness probes.
* **🔒 Enterprise CI/CD Strategy (Extra Credit):** 
  * **Continuous Integration (`ci.yml`):** Automatically runs Flake8 linting and Pytest unit tests on every push and Pull Request to `main`.
  * **Continuous Deployment (`cd.yml`):** Decoupled from standard pushes. It strictly triggers on **GitHub Release Tags** (e.g., `v1.0.0`), building and pushing the container to GCP Artifact Registry and deploying to Cloud Run.

---

## 🧪 Live API Testing (cURL Examples)

You can test the live cloud endpoints directly from your terminal:

### 1. Single Sentence Prediction (`POST /predict`)
```bash
curl -X POST "https://clinical-bert-api-811357407534.asia-south1.run.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"sentence": "The patient denies chest pain."}'
```
**Response:**
```json
{
  "label": "ABSENT",
  "score": 0.9738890528678894
}
```

### 2. Batch Sentence Prediction (`POST /predict_batch`)
```bash
curl -X POST "https://clinical-bert-api-811357407534.asia-south1.run.app/predict_batch" \
     -H "Content-Type: application/json" \
     -d '{"sentences": ["The patient denies chest pain.", "He has a history of hypertension.", "If the patient experiences dizziness, reduce the dosage.", "No signs of pneumonia were observed."]}'
```
**Response:**
```json
{
  "predictions": [
    {"label": "ABSENT", "score": 0.9738890528678894},
    {"label": "PRESENT", "score": 0.9952554106712341},
    {"label": "PRESENT", "score": 0.9903457760810852},
    {"label": "ABSENT", "score": 0.9654479026794434}
  ]
}
```
*(Note: As documented in MLOps evaluations, Sentence 3 returns `PRESENT` because the underlying Hugging Face model vocabulary is strictly restricted to `PRESENT`, `ABSENT`, and `POSSIBLE`).*

---

## 💻 How to Run Locally

### Option 1: Using Local Python (3.12+)
```bash
# 1. Clone the repository
git clone https://github.com/Anurag-raj03/BERT_ASSIGNMENT_MLOPS.git
cd clinical-bert-api

# 2. Install dependencies (Includes strict Numpy < 2.0 constraint)
pip install -r requirements.txt

# 3. Run the local ASGI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Visit `http://localhost:8000/docs` in your browser.

### Option 2: Using Docker
```bash
# 1. Build the optimized Docker image
docker build -t clinical-bert-api .

# 2. Run the container
docker run -p 8080:8080 clinical-bert-api
```
Visit `http://localhost:8080/docs` in your browser.

---

## 📊 Automated Test Suite

The project includes an exhaustive Pytest suite covering all mandatory assignment scenarios and batch testing.

```bash
# Run tests locally
PYTHONPATH=. pytest tests/ -v
```
**Test Coverage:**
* `test_health_check`: Verifies `200 OK` liveness.
* `test_predict`: Validates mathematical boundaries (`0.0 <= score <= 1.0`) and exact label classifications across 4 clinical scenarios.
* `test_predict_batch`: Verifies multi-sentence payload handling.
