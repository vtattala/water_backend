from typing import Optional
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ✅ FIX: load model WITHOUT compile (prevents Keras version crash on Render)
model = tf.keras.models.load_model("groundwater_model.keras", compile=False)

# Model input shape constants
TIME_STEPS = 6
GRID_HEIGHT = 32
GRID_WIDTH = 36
CHANNELS = 1


# ✅ Request format (this matches your Android app)
class PredictionRequest(BaseModel):
    region: str
    time_range: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# --- Helper functions ---

def normalize_region(region: str) -> str:
    return region.strip().lower().replace(" ", "_")


def build_input_tensor(region: str, time_range: str) -> np.ndarray:
    """
    TEMPORARY: Generates input for model
    (Later: replace with real satellite data pipeline)
    """

    region_key = normalize_region(region)

    region_base_map = {
        "california": 0.62,
        "michigan": 0.48,
    }

    time_range_factor_map = {
        "6_months": 0.04,
        "1_year": -0.03,
        "custom_range": 0.01,
    }

    base_level = region_base_map.get(region_key, 0.50)
    time_adjustment = time_range_factor_map.get(time_range, 0.0)

    tensor = np.zeros((1, TIME_STEPS, GRID_HEIGHT, GRID_WIDTH, CHANNELS), dtype=np.float32)

    x_gradient = np.linspace(-0.08, 0.08, GRID_WIDTH, dtype=np.float32)
    y_gradient = np.linspace(0.06, -0.06, GRID_HEIGHT, dtype=np.float32).reshape(GRID_HEIGHT, 1)

    for t in range(TIME_STEPS):
        seasonal_shift = (t - (TIME_STEPS - 1) / 2.0) * 0.015
        slice_values = base_level + time_adjustment + seasonal_shift + y_gradient + x_gradient
        tensor[0, t, :, :, 0] = np.clip(slice_values, 0.0, 1.0)

    return tensor


def prediction_to_heatmap(prediction: np.ndarray):
    flat = prediction.reshape(-1)

    # Ensure correct size
    target_size = GRID_HEIGHT * GRID_WIDTH
    if flat.size < target_size:
        flat = np.tile(flat, int(np.ceil(target_size / flat.size)))

    flat = flat[:target_size]
    return flat.reshape(GRID_HEIGHT, GRID_WIDTH).tolist()


def summarize_prediction(heatmap):
    avg = float(np.mean(heatmap))

    if avg < 0.33:
        return "Low"
    elif avg > 0.66:
        return "High"
    else:
        return "Normal"


# --- Routes ---

@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # ✅ Build model input
        model_input = build_input_tensor(request.region, request.time_range)

        # ✅ Run prediction
        pred = model.predict(model_input, verbose=0)

        # ✅ Convert to usable output
        heatmap = prediction_to_heatmap(pred)
        summary = summarize_prediction(heatmap)

        return {
            "region": request.region,
            "time_range": request.time_range,
            "groundwater_level_status": summary,
            "heatmap": heatmap,
            "model_input_shape": list(model_input.shape),
        }

    except Exception as e:
        return {"error": str(e)}
