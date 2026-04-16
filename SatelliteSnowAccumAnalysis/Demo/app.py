from __future__ import annotations

from base64 import b64encode
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from PIL import Image

from cnn_segmentation import DEFAULT_MODEL, load_session, prepare_input_tensor


APP_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = APP_DIR / "public"
PUBLIC_SAMPLE_DIR = PUBLIC_DIR / "sample-images"

DEMO_CITIES = {
    "boston": {
        "label": "Boston",
        "samples": [
            "boston__wy2017__S2A_19TCG_20180118_0_L2A__y0000_x0000",
            "boston__wy2017__S2A_19TCG_20180118_0_L2A__y0000_x0256",
            "boston__wy2017__S2A_19TCG_20180118_0_L2A__y0000_x0512",
            "boston__wy2017__S2A_19TCG_20180118_0_L2A__y0000_x0768",
            "boston__wy2017__S2A_19TCG_20180118_0_L2A__y0000_x1024",
        ],
    },
    "buffalo": {
        "label": "Buffalo",
        "samples": [
            "buffalo__wy2016__S2A_17TPH_20170218_0_L2A__y0256_x0256",
            "buffalo__wy2016__S2A_17TPH_20170218_0_L2A__y0512_x1024",
            "buffalo__wy2016__S2A_17TPH_20170218_0_L2A__y0768_x1024",
            "buffalo__wy2016__S2A_17TPH_20170218_0_L2A__y1024_x0512",
            "buffalo__wy2016__S2A_17TPH_20170218_0_L2A__y1024_x1024",
        ],
    },
    "chicago": {
        "label": "Chicago",
        "samples": [
            "chicago__wy2017__S2A_16TDM_20180225_0_L2A__y0000_x0512",
            "chicago__wy2017__S2A_16TDM_20180225_0_L2A__y0000_x1792",
            "chicago__wy2017__S2A_16TDM_20180225_0_L2A__y0512_x1280",
            "chicago__wy2017__S2A_16TDM_20180225_0_L2A__y0768_x0512",
            "chicago__wy2017__S2A_16TDM_20180225_0_L2A__y1280_x0512",
        ],
    },
    "new_york": {
        "label": "New York",
        "samples": [
            "new_york__wy2016__S2A_18TWK_20170116_0_L2A__y3072_x0256",
            "new_york__wy2016__S2A_18TWK_20170116_0_L2A__y3072_x1280",
            "new_york__wy2016__S2A_18TWK_20170116_0_L2A__y3072_x2048",
            "new_york__wy2016__S2A_18TWK_20170116_0_L2A__y3072_x2304",
            "new_york__wy2016__S2A_18TWK_20170116_0_L2A__y3072_x2560",
        ],
    },
}

app = FastAPI(title="Snow Accum Analysis Segmentation Demo")


@lru_cache(maxsize=1)
def get_session():
    if not DEFAULT_MODEL.exists():
        raise RuntimeError(f"Model file not found: {DEFAULT_MODEL}")
    return load_session(DEFAULT_MODEL)


def png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


def build_catalog() -> list[dict]:
    cities = []
    for city_id, city in DEMO_CITIES.items():
        samples = [
            {
                "id": sample_id,
                "label": f"Sample {index}",
                "image_url": f"/sample-images/{sample_id}/image.png",
            }
            for index, sample_id in enumerate(city["samples"], start=1)
        ]
        cities.append(
            {
                "id": city_id,
                "label": city["label"],
                "samples": samples,
            }
        )
    return cities


def snow_probability_from_output(output: np.ndarray) -> np.ndarray:
    prediction = np.asarray(output)

    if prediction.ndim == 4 and prediction.shape[:2] == (1, 2):
        logits_neg = prediction[0, 0]
        logits_pos = prediction[0, 1]
        return (1.0 / (1.0 + np.exp(-(logits_pos + logits_neg)))).astype(np.float32)
    if prediction.ndim == 4 and prediction.shape[:2] == (1, 1):
        logits = prediction[0, 0]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        logits = prediction[0]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    if prediction.ndim == 2:
        return (1.0 / (1.0 + np.exp(-prediction))).astype(np.float32)

    raise HTTPException(
        status_code=500,
        detail=f"Model output had an unexpected shape: {list(prediction.shape)}",
    )


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/index.html", status_code=307)


@app.head("/", include_in_schema=False)
def home_head() -> Response:
    return Response(status_code=200)


@app.get("/app", include_in_schema=False)
def demo_page() -> RedirectResponse:
    return RedirectResponse(url="/app.html", status_code=307)


@app.head("/app", include_in_schema=False)
def demo_page_head() -> Response:
    return Response(status_code=200)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_present": DEFAULT_MODEL.exists(),
        "public_present": PUBLIC_DIR.exists(),
        "sample_images_present": PUBLIC_SAMPLE_DIR.exists(),
    }


@app.get("/demo-config")
def demo_config() -> JSONResponse:
    cities = build_catalog()
    return JSONResponse(
        {
            "cities": cities,
            "default_city": cities[0]["id"] if cities else None,
        }
    )


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    filename = image.filename or "uploaded-image.png"
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="No image data was uploaded.")

    try:
        input_tensor, visual_tensor, meta = prepare_input_tensor(raw_bytes, filename)
        session = get_session()
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_tensor})[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    snow_probability = snow_probability_from_output(output)
    snow_fraction = float(snow_probability.mean())

    visual = np.asarray(visual_tensor).squeeze()
    rgb = np.moveaxis(visual[:3], 0, -1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    heatmap = np.full((*snow_probability.shape, 3), 255.0, dtype=np.float32)
    heatmap[..., 1] = 255.0 * (1.0 - snow_probability)
    heatmap[..., 2] = 255.0 * (1.0 - snow_probability)
    heatmap = heatmap.astype(np.uint8)

    overlay = (0.45 * rgb + 0.55 * heatmap).astype(np.uint8)

    return JSONResponse(
        {
            "filename": filename,
            "snow_fraction": snow_fraction,
            "mask_shape": list(snow_probability.shape),
            "prediction_mask_base64": png_base64(Image.fromarray(heatmap, mode="RGB")),
            "overlay_base64": png_base64(Image.fromarray(overlay, mode="RGB")),
            "meta": meta,
        }
    )
