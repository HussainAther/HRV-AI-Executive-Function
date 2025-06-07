import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# --- Load and preprocess HRV data ---
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    # Use 3 key HRV features
    X = df[['HRV_RMSSD', 'HRV_SDNN', 'HRV_LF_HF']].values
    y = df['ExecutiveFunctionScore'].values  # Assume a normalized score [0–1]

    # Normalize input features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))  # LSTM expects 3D input

    return X_reshaped, y

# --- Build LSTM model ---
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Output score between 0 and 1
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# --- Train model and save to /models ---
def train_and_save_model(csv_path, model_output_path):
    df = load_data(csv_path)
    X, y = prepare_features(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((X.shape[1], 1))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=16)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    print(f"✅ Model trained and saved to {model_output_path}")

# --- Entry point ---
if __name__ == "__main__":
    train_and_save_model("../data/sample_hrv_data.csv", "../models/hrv_lstm_model.h5")

