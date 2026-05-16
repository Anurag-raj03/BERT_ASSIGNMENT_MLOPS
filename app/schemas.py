from pydantic import BaseModel

class PredictRequest(BaseModel):
    sentence: str

class PredictResponse(BaseModel):
    label: str
    score: float
