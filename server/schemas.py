from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_hazard: bool
    probabilities: dict[str, float]
    model_name: str
    latency_ms: float


class PredictAllResponse(BaseModel):
    results: dict[str, PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_name: str
