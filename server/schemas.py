from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_hazard: bool
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_name: str
