from fastapi import FastAPI
import numpy as np
import tensorflow as tf

app = FastAPI()

# ✅ Load model once (on startup)
model = tf.keras.models.load_model("groundwater_model.keras")

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: dict):
    try:
        x = np.array(data["input"], dtype=np.float32)

        # 🔥 FORCE CORRECT SHAPE
        x = x.reshape(1, 6, 32, 36, 1)

        pred = model.predict(x)

        return {"prediction": pred.tolist()}

    except Exception as e:
        return {"error": str(e)}