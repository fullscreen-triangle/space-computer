import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# LLM settings
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", str(MODELS_DIR / "biomech_llm"))
LLM_QUANTIZATION = os.getenv("LLM_QUANTIZATION", "q4_0")  # 4-bit quantization
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Video processing
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "avi"]
TEMP_UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
