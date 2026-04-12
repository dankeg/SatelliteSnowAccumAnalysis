from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.responses import RedirectResponse
import torch

app = FastAPI()

# Load once at import time for a tiny demo.
# Replace this with your real model class / loading code.
class DummyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(20, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)

model = DummyCNN()
model.eval()

# Example:
# state = torch.load("model/model.pt", map_location="cpu")
# model.load_state_dict(state)
# model.eval()

class PredictRequest(BaseModel):
    series: list[float]

@app.get("/", include_in_schema=False)
async def home():
    return RedirectResponse("/index.html", status_code=307)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictRequest):
    arr = np.array(req.series, dtype=np.float32)

    # Example shape handling for a small 1D time-series window of length 20.
    # Adjust to match your real CNN input shape.
    if arr.shape[0] != 20:
        return {
            "error": "Expected exactly 20 timesteps.",
            "received_length": int(arr.shape[0]),
        }

    x = torch.tensor(arr, dtype=torch.float32).view(1, 1, 20)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }