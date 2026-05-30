from typing import Dict
from app_backend.config import Settings


def explain(class_index: int, confidence: float, settings: Settings) -> Dict[str, str]:
	"""Return a simple explanation/advice given the predicted class index and confidence.

	This implementation uses the `settings.class_names` list to map indices to labels and
	returns a short advice string. It can be replaced by more sophisticated explainers.
	"""
	try:
		label = settings.class_names[class_index]
	except Exception:
		label = "unknown"

	if label.lower() in ("malignant", "cancer", "suspicious"):
		advice = "High risk — seek dermatologist evaluation promptly."
	elif label.lower() in ("benign", "normal"):
		advice = "Low risk — monitor and follow routine skin checks."
	else:
		advice = "Consult a medical professional for assessment."

	return {
		"label": label,
		"advice": advice,
	}


__all__ = ["explain"]

