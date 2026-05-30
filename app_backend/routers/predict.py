from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from app_backend.models import ml_models
from app_backend.services import preprocessor, explainer
from app_backend.models.schemas import PredictionResponse, ErrorResponse

import numpy as np

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict(file: UploadFile = File(...), request: Request = None):
	settings = request.app.state.settings
	try:
		batch = await preprocessor.preprocess_upload(file, settings)
		model = ml_models.get_model(settings)
		preds = model.predict(batch)
		probs = np.asarray(preds[0])
		class_index = int(np.argmax(probs))
		confidence = float(probs[class_index])
		explanation = explainer.explain(class_index, confidence, settings)
		return PredictionResponse(label=explanation["label"], class_index=class_index, confidence=confidence, advice=explanation["advice"])
	except HTTPException as e:
		raise e
	except Exception as e:
		return JSONResponse(status_code=500, content={"detail": str(e)})


__all__ = ["router"]

