from typing import Optional
from pathlib import Path

import numpy as np


try:
    import tf_keras
    load_model = tf_keras.models.load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        from tensorflow.keras.models import load_model

from app_backend.config import Settings


_model: Optional[object] = None


def initialize(settings: Settings):
	"""Load the model into memory (singleton)."""
	global _model
	if _model is not None:
		return _model

	model_path = Path(settings.model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	if load_model is None:
		raise RuntimeError("TensorFlow is not available (cannot load model)")

	_model = load_model(str(model_path))
	# Warm up the model with a dummy batch
	dummy = np.zeros((1, settings.input_size, settings.input_size, 3), dtype=np.float32)
	try:
		_model.predict(dummy)
	except Exception:
		# ignore warmup errors but keep the model reference
		pass

	return _model


def get_model(settings: Optional[Settings] = None):
	if _model is None:
		if settings is None:
			raise RuntimeError("Model not initialized. Call initialize(settings) on startup.")
		return initialize(settings)
	return _model


__all__ = ["initialize", "get_model"]

