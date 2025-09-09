from fastapi import APIRouter, HTTPException
from datetime import datetime
from pydantic import BaseModel
from typing import List
import json

router = APIRouter()

# Mock in-memory user progress (replace with DB logic)
user_progress = {
    "xp": 0,
    "level": 1,
    "streak": 0,
    "last_logged_date": None,
    "achievements": [],
    "quests": []
}

# -----------------------------
# Models
# -----------------------------
class XPUpdate(BaseModel):
    amount: int

class HabitLog(BaseModel):
    habit: str
    timestamp: str

class Achievement(BaseModel):
    name: str
    description: str

class Quest(BaseModel):
    title: str
    description: str
    complete: bool

# -----------------------------
# Routes
# -----------------------------
@router.post("/add-xp")
def add_xp(data: XPUpdate):
    user_progress["xp"] += data.amount
    if user_progress["xp"] >= user_progress["level"] * 100:
        user_progress["level"] += 1
        user_progress["xp"] = 0
    return user_progress

@router.post("/log-habit-streak")
def log_streak(entry: HabitLog):
    today = datetime.utcnow().date()
    last_date = user_progress["last_logged_date"]

    if last_date is None or (today - last_date).days > 1:
        user_progress["streak"] = 1
    elif (today - last_date).days == 1:
        user_progress["streak"] += 1

    user_progress["last_logged_date"] = today
    return {"streak": user_progress["streak"]}

@router.get("/get-streak")
def get_streak():
    return {"streak": user_progress["streak"]}

@router.post("/add-achievement")
def add_achievement(ach: Achievement):
    user_progress["achievements"].append(ach.dict())
    return {"achievements": user_progress["achievements"]}

@router.get("/get-achievements")
def get_achievements():
    return {"achievements": user_progress["achievements"]}

@router.post("/add-quest")
def add_quest(q: Quest):
    user_progress["quests"].append(q.dict())
    return {"quests": user_progress["quests"]}

@router.get("/get-quests")
def get_quests():
    return {"quests": user_progress["quests"]}

