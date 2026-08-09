"""FastAPI service exposing a malaria cell-image classifier."""

import io
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from model import CLASS_NAMES, IMG_SIZE, MODEL_PATH, build_model

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_VERSION = "v1.0.0"

ml = {}


def _self_check(model) -> bool:
    """Probe the model with synthetic inputs to detect untrained weights.

    An untrained network returns almost the same score for every input, so a
    near-zero spread means the weights carry no information.
    """
    probes = np.stack(
        [
            np.zeros((IMG_SIZE, IMG_SIZE, 3), np.float32),
            np.ones((IMG_SIZE, IMG_SIZE, 3), np.float32),
            np.full((IMG_SIZE, IMG_SIZE, 3), 0.5, np.float32),
        ]
    )
    scores = model.predict(probes, verbose=0).ravel()
    spread = float(scores.max() - scores.min())

    if spread < 0.15:
        print("=" * 68)
        print("WARNING: model appears UNTRAINED (output spread "
              f"{spread:.4f}).")
        print("Predictions will be meaningless - everything will collapse")
        print("to a single class. Train it with:")
        print("    python train.py --data-dir dataset")
        print("=" * 68)
        return False
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        ml["model"] = keras.models.load_model(MODEL_PATH)
        stamp = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        print(f"Loaded model from {MODEL_PATH}")
        print(f"  weights modified: {stamp:%Y-%m-%d %H:%M:%S}")
    else:
        ml["model"] = build_model()
        print(f"{MODEL_PATH} not found - using a freshly initialized model.")
        print("Run `python model.py` to create the weights file.")

    ml["trained"] = _self_check(ml["model"])
    yield
    ml.clear()


app = FastAPI(
    title="Malaria Detection API",
    description="Binary classification of blood-smear cell images.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(raw: bytes) -> np.ndarray:
    """Decode bytes to a normalized (1, 128, 128, 3) float32 batch."""
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_loaded": ml.get("model") is not None,
        "model_trained": ml.get("trained", False),
        "model_version": MODEL_VERSION,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use .jpg, .jpeg or .png.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    start = time.perf_counter()
    batch = preprocess(raw)
    score = float(ml["model"].predict(batch, verbose=0)[0][0])
    elapsed_ms = (time.perf_counter() - start) * 1000

    is_parasitized = score >= 0.5
    confidence = score if is_parasitized else 1.0 - score

    return {
        "prediction": CLASS_NAMES[1] if is_parasitized else CLASS_NAMES[0],
        "confidence": round(confidence * 100, 2),
        "processing_time_ms": round(elapsed_ms, 2),
        "model_version": MODEL_VERSION,
        "model_trained": ml.get("trained", False),
    }
