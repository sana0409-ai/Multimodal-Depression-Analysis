# src/offline_pipeline.py

from pathlib import Path
import pandas as pd
import joblib
import torch
import numpy as np
from torch import nn

def build_model(input_dim: int, hidden_units=32, dropout=0.2):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.BatchNorm1d(hidden_units),
        nn.LeakyReLU(0.1),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, hidden_units),
        nn.BatchNorm1d(hidden_units),
        nn.LeakyReLU(0.1),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, 2),
    )

def main():
    base = Path(__file__).resolve().parent.parent

    # Load combined CSV
    df = pd.read_csv(base / 'data' / 'daic_patient_level_multimodal.csv')
    print("Loaded CSV with shape:", df.shape)

    # 2) Pull out only the top‐100 features + labels
    top_feats = joblib.load(base / 'models' / 'top100_features.joblib')
    X = df[top_feats].values            # shape (n_samples, 100)
    y = df['label'].values.astype(int)  # shape (n_samples,)

    # 3) Scale them 
    scaler = joblib.load(base / 'models' / 'scaler.joblib')
    X_scaled = scaler.transform(X)

    # 4) Rebuild the exact same architecture and load the weights
    model = build_model(input_dim=X_scaled.shape[1])
    state = torch.load(base / 'models' / 'depression_model.pt', map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # 5) Run inference on all samples
    with torch.no_grad():
        inputs = torch.from_numpy(X_scaled).float()
        outputs = model(inputs)                     # (n_samples, 2)
        probs   = torch.softmax(outputs, dim=1)[:,1] # P(depressed)
        preds   = (probs > 0.5).numpy().astype(int)

    # 6) Print a sample & overall accuracy
    print(f"\nSample 0 →  P(depressed)={probs[0]:.2f}, pred={preds[0]}, true={y[0]}")
    acc = (preds == y).mean()
    print(f"Overall accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()
