from fastapi import FastAPI
from app_backend.config import Settings
from app_backend.models import ml_models
from app_backend.routers import predict as predict_router


def create_app() -> FastAPI:
	app = FastAPI(title="Skin Disease Classifier")
	# Load settings once at startup (Path A)
	settings = Settings()
	app.state.settings = settings

	@app.on_event("startup")
	async def startup_event():
		# Initialize and warm the ML model into memory (singleton)
		try:
			ml_models.initialize(settings)
		except Exception as e:
			import logging
			logging.warning("Model initialization failed at startup: %s", e)

	app.include_router(predict_router.router)
	return app


app = create_app()

