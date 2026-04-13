from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image

try:
    from .cnn_segmentation import (
        DEFAULT_MODEL,
        DEFAULT_TARGET_SIZE,
        load_session,
        prepare_input_tensor,
    )
except ImportError:
    APP_DIR = Path(__file__).resolve().parent
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from cnn_segmentation import (
        DEFAULT_MODEL,
        DEFAULT_TARGET_SIZE,
        load_session,
        prepare_input_tensor,
    )


APP_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = APP_DIR / "public"

app = FastAPI(title="Snow Segmentation Demo")

_session = None


def get_session():
    global _session
    if _session is None:
        _session = load_session(DEFAULT_MODEL)
    return _session


def _tensor_to_mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.squeeze().astype(np.uint8) * 255), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


def _tensor_to_overlay_png(rgb_tensor: np.ndarray, mask: np.ndarray) -> str:
    rgb = np.clip(rgb_tensor[:3], 0.0, 1.0)
    base = Image.fromarray((np.moveaxis(rgb, 0, -1) * 255).astype(np.uint8), mode="RGB")
    mask_img = Image.fromarray((mask.squeeze().astype(np.uint8) * 255), mode="L")
    width, height = base.size
    white = Image.new("RGB", (width, height), (255, 255, 255))
    image = Image.composite(white, base, mask_img)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


@app.get("/", include_in_schema=False)
async def home():
    return RedirectResponse("/index.html", status_code=307)


@app.get("/app.js", include_in_schema=False)
async def app_js():
    return FileResponse(PUBLIC_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
def health():
    return {
        "ok": True,
        "checkpoint": str(DEFAULT_MODEL.name),
        "session_loaded": _session is not None,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        input_tensor, visual_tensor, meta = prepare_input_tensor(
            raw,
            image.filename or "upload.png",
            DEFAULT_TARGET_SIZE,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        session = get_session()
        logits = session.run(["logits"], {"input": input_tensor})[0]
        mask = np.argmax(logits, axis=1).astype(np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    snow_pixels = int((mask == 1).sum())
    total_pixels = int(mask.size)
    snow_fraction = snow_pixels / total_pixels if total_pixels else 0.0

    return {
        "prediction_mask_base64": _tensor_to_mask_png(mask),
        "overlay_base64": _tensor_to_overlay_png(visual_tensor.squeeze(0), mask),
        "snow_fraction": snow_fraction,
        "snow_pixels": snow_pixels,
        "total_pixels": total_pixels,
        "meta": meta,
    }
