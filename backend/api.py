from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel
import uvicorn
import asyncio
import csv
import io

app = FastAPI()

LEADERBOARD_FILE = "data/reaction_leaderboard.csv"

class ReactionEntry(BaseModel):
    name: str
    reaction_time_ms: int

@app.post("/log-reaction")
async def log_reaction(entry: ReactionEntry):
    os.makedirs("data", exist_ok=True)
    with open(LEADERBOARD_FILE, "a") as f:
        writer = csv.writer(f)
        writer.writerow([entry.name, entry.reaction_time_ms])
    return {"status": "logged"}

@app.get("/leaderboard")
async def get_leaderboard():
    if not os.path.exists(LEADERBOARD_FILE):
        return []

    with open(LEADERBOARD_FILE, "r") as f:
        rows = list(csv.reader(f))
        rows = sorted(rows, key=lambda x: int(x[1]))
        top_10 = [{"name": row[0], "reaction_time_ms": int(row[1])} for row in rows[:10]]
    return top_10


# Load pre-trained HRV model
MODEL_PATH = "../models/hrv_ai_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

STREAK_LOG_PATH = "streak_log.json"

class StreakEntry(BaseModel):
    streak: int
    max_streak: int
    timestamp: datetime
    session_id: str = "default"


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

@app.post("/log-streak")
def log_streak(entry: StreakEntry):
    log = []
    if os.path.exists(STREAK_LOG_PATH):
        with open(STREAK_LOG_PATH, "r") as f:
            log = json.load(f)

    log.append(entry.dict())

    with open(STREAK_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return {"status": "ok", "entry_logged": entry}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

