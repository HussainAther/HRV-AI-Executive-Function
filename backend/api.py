from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
from datetime import datetime
import uvicorn

app = FastAPI()

MODEL_PATH = "../models/hrv_lstm_model.h5"
LOG_PATH = "../data/prediction_log.json"

model = tf.keras.models.load_model(MODEL_PATH)

class HRVInput(BaseModel):
    HRV_RMSSD: float
    HRV_SDNN: float
    HRV_LF_HF: float

@app.post("/predict")
def predict(hrv: HRVInput):
    try:
        input_array = np.array([[hrv.HRV_RMSSD, hrv.HRV_SDNN, hrv.HRV_LF_HF]])
        input_array = np.reshape(input_array, (1, 3, 1))
        prediction = model.predict(input_array)[0][0]

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "HRV_RMSSD": hrv.HRV_RMSSD,
            "HRV_SDNN": hrv.HRV_SDNN,
            "HRV_LF_HF": hrv.HRV_LF_HF,
            "executive_function_score": float(prediction)
        }

        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(result)
        with open(LOG_PATH, "w") as f:
            json.dump(logs, f, indent=2)

        return {"executive_function_score": round(float(prediction), 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

