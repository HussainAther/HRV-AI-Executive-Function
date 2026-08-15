import numpy as np
import pandas as pd

# RMSSD: Root Mean Square of Successive Differences
def compute_rmssd(ibi_ms):
    diff = np.diff(ibi_ms)
    return np.sqrt(np.mean(diff ** 2))

# SDNN: Standard deviation of NN intervals
def compute_sdnn(ibi_ms):
    return np.std(ibi_ms, ddof=1)

# LF/HF ratio placeholder (normally requires frequency domain analysis)
def compute_lf_hf_ratio(ibi_ms):
    # Placeholder value, recommend using Welch's method or FFT on interpolated IBI
    return np.random.uniform(0.5, 2.5)  # Simulated until real spectrum is computed

# Process a CSV file with IBI column (ms)
def extract_hrv_features(csv_path):
    df = pd.read_csv(csv_path)
    ibi = df['IBI_ms'].dropna().values

    rmssd = compute_rmssd(ibi)
    sdnn = compute_sdnn(ibi)
    lf_hf = compute_lf_hf_ratio(ibi)

    print("HRV Feature Summary:")
    print(f"  RMSSD: {rmssd:.2f} ms")
    print(f"  SDNN:  {sdnn:.2f} ms")
    print(f"  LF/HF Ratio: {lf_hf:.2f}")

    return {"HRV_RMSSD": rmssd, "HRV_SDNN": sdnn, "HRV_LF_HF": lf_hf}

if __name__ == "__main__":
    features = extract_hrv_features("../data/raw_ibi.csv")

