from fastapi import UploadFile, HTTPException
try:
	from PIL import Image
except Exception:
	Image = None
import io
import numpy as np
from typing import Tuple
from app_backend.config import Settings


async def preprocess_upload(file: UploadFile, settings: Settings) -> np.ndarray:
	if not file.content_type or not file.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="Uploaded file is not an image")
	data = await file.read()
	return preprocess_bytes(data, settings)


def preprocess_bytes(data: bytes, settings: Settings) -> np.ndarray:
	if Image is None:
		raise HTTPException(status_code=500, detail="Pillow is required. Install with: pip install Pillow")
	try:
		img = Image.open(io.BytesIO(data)).convert("RGB")
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid image data")

	img = img.resize((settings.input_size, settings.input_size))
	arr = np.asarray(img, dtype=np.float32) / 255.0
	if arr.ndim == 2:
		arr = np.stack([arr] * 3, axis=-1)
	if arr.shape[-1] != 3:
		raise HTTPException(status_code=400, detail="Image must have 3 channels")
	return np.expand_dims(arr, axis=0)


def validate_shape(batch: np.ndarray, settings: Settings) -> Tuple[int, int, int, int]:
	"""Return the shape of the preprocessed batch and validate it matches settings."""
	if batch.ndim != 4:
		raise ValueError("Preprocessed batch must be 4D (B,H,W,C)")
	b, h, w, c = batch.shape
	if h != settings.input_size or w != settings.input_size:
		raise ValueError("Preprocessed image size does not match settings.input_size")
	return batch.shape


__all__ = ["preprocess_upload", "preprocess_bytes", "validate_shape"]

