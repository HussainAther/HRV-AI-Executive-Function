import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset (placeholder, replace with actual data source)
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare dataset for training
def prepare_data(df):
    features = df[['HRV_RMSSD', 'HRV_SDNN', 'HRV_LF_HF']].values
    labels = df['Cognitive_Performance'].values
    return features, labels

# Build LSTM model for HRV-based executive function prediction
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example usage
    df = load_data("../data/sample_hrv_data.csv")
    X, y = prepare_data(df)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshaping for LSTM input
    
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)
    model.save("../models/hrv_ai_model.h5")

