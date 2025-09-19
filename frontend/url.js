// frontend/url.js

const BASE_URL = "http://localhost:8000"; // adjust if needed for deployment

const API = {
  PREDICT: `${BASE_URL}/predict`,
  LOG_HABIT: `${BASE_URL}/log-habit`,
  GET_HABITS: `${BASE_URL}/get-habits`,
  STREAKS: `${BASE_URL}/streaks`,
  LEADERBOARD: `${BASE_URL}/leaderboard`,
  PREDICTION_LOG: `${BASE_URL}/api/prediction-log`,
};

export default API;

