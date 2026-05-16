from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time

from app.schemas import PredictRequest, PredictResponse
from app.model import bert_classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    bert_classifier.load_model()
    yield
    # Clean up on shutdown
    pass

app = FastAPI(
    title="Clinical BERT Real-Time Inference API",
    description="API for clinical assertion/negation text classification.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Start timing for telemetry if needed
        start_time = time.time()
        
        prediction = bert_classifier.predict(request.sentence)
        
        end_time = time.time()
        print(f"Prediction for '{request.sentence}' took {end_time - start_time:.4f}s")
        
        return PredictResponse(
            label=prediction["label"],
            score=prediction["score"]
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
