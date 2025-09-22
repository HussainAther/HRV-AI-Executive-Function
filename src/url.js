// src/url.js
import config from "@/config";

const API = {
  PREDICT: `${config.API_BASE_URL}/predict`,
  LOG_HABIT: `${config.API_BASE_URL}/log-habit`,
  GET_HABITS: `${config.API_BASE_URL}/get-habits`,
  STREAKS: `${config.API_BASE_URL}/streaks`,
  LEADERBOARD: `${config.API_BASE_URL}/leaderboard`,
  PREDICTION_LOG: `${config.API_BASE_URL}/api/prediction-log`,
};

export default API;

