import os
from pathlib import Path
from typing import List

from pydantic import BaseSettings  # BaseSettings is built into pydantic v1

MODEL_PATH = Path(__file__).resolve().parent / "pretrained_model" / "modelv1.h5"
CLASS_NAMES = ["benign", "malignant"]
INPUT_SIZE = 28
DEBUG = False
HOST = "0.0.0.0"
PORT = 8000
ENV_PREFIX = "SKIN_"


class Settings(BaseSettings):
    model_path: Path = MODEL_PATH
    input_size: int = INPUT_SIZE
    class_names: List[str] = CLASS_NAMES
    debug: bool = DEBUG
    host: str = HOST
    port: int = PORT

    class Config:
        env_prefix = ENV_PREFIX


__all__ = ["Settings"]