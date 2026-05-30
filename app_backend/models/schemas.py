from pydantic import BaseModel


class PredictionResponse(BaseModel):
	label: str
	class_index: int
	confidence: float
	advice: str


class ErrorResponse(BaseModel):
	detail: str


__all__ = ["PredictionResponse", "ErrorResponse"]

