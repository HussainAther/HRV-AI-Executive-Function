from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel
import uvicorn
import asyncio

app = FastAPI()

# Load pre-trained HRV model
MODEL_PATH = "../models/hrv_ai_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class HRVInput(BaseModel):
    HRV_RMSSD: float
    HRV_SDNN: float
    HRV_LF_HF: float

@app.post("/predict")
def predict(hrv_data: HRVInput):
    try:
        features = np.array([[hrv_data.HRV_RMSSD, hrv_data.HRV_SDNN, hrv_data.HRV_LF_HF]])
        features = np.reshape(features, (features.shape[0], features.shape[1], 1))  # Reshape for LSTM
        prediction = model.predict(features)[0][0]
        
        return {"executive_function_score": round(float(prediction), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-hrv-data")
def upload_data(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_csv("../data/user_uploaded_hrv_data.csv", index=False)
        return {"message": "File uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # Simulate live HRV processing (Replace with actual real-time data handling)
        response = {"message": "Live HRV data received", "data": data}
        await websocket.send_json(response)
        await asyncio.sleep(1)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

