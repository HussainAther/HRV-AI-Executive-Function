import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# --- Load model ---
model_path = '../models/hrv_lstm_model.h5'
model = tf.keras.models.load_model(model_path)

# --- Load data ---
data_path = '../data/sample_hrv_data.csv'
df = pd.read_csv(data_path)

# --- Prepare features and labels ---
X = df[['HRV_RMSSD', 'HRV_SDNN', 'HRV_LF_HF']].values
y_true = df['ExecutiveFunctionScore'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_input = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

# --- Predict ---
y_pred = model.predict(X_input).flatten()

# --- Evaluate ---
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f'\n\U0001F389 Evaluation complete!')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')

# --- Save results ---
output_df = df.copy()
output_df['PredictedScore'] = y_pred
output_path = '../data/predictions.csv'
output_df.to_csv(output_path, index=False)
print(f'\U0001F4C5 Predictions saved to {output_path}')

# --- Optional plot ---
plt.figure(figsize=(10, 5))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('Executive Function Score: True vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig('../data/evaluation_plot.png')
print('\U0001F5BC Saved plot to ../data/evaluation_plot.png')

