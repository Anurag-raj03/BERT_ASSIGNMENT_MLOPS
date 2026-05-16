from typing import List
from pydantic import BaseModel

class PredictRequest(BaseModel):
    sentence: str

class PredictResponse(BaseModel):
    label: str
    score: float

class BatchPredictRequest(BaseModel):
    sentences: List[str]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
