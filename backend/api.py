from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI()

# Load the trained model
MODEL_PATH = "../models/hrv_lstm_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Input schema for prediction
class HRVInput(BaseModel):
    HRV_RMSSD: float
    HRV_SDNN: float
    HRV_LF_HF: float

@app.post("/predict")
def predict(hrv: HRVInput):
    try:
        input_array = np.array([[hrv.HRV_RMSSD, hrv.HRV_SDNN, hrv.HRV_LF_HF]])
        input_array = np.reshape(input_array, (1, 3, 1))  # LSTM expects 3D input
        prediction = model.predict(input_array)[0][0]
        return {
            "executive_function_score": round(float(prediction), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-hrv-data")
def upload_data(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_csv(f"../data/{file.filename}", index=False)
        return {
            "message": "HRV data uploaded successfully",
            "rows": len(df),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: run the API locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

