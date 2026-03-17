# pyre-ignore-all-errors
"""
Train the lung cancer risk prediction model and save it.
Run this script once to generate models/lung_cancer_model.pkl

Option A – Real data (preferred):
  1. Download the dataset from Kaggle:
     https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link
  2. Place "cancer patient data sets.csv" in the project root (next to this script).
  3. Run: python train_lung_model.py

Option B – Synthetic data (demo, used when CSV is not found):
  The script will generate a synthetic dataset that reproduces the statistical
  properties of the original data so the API still works end-to-end.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Look for the CSV in a few common places
_CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "cancer patient data sets.csv"),
    os.path.join(BASE_DIR, "cancer-patients-and-air-pollution-a-new-link", "cancer patient data sets.csv"),
    os.path.join(BASE_DIR, "data", "cancer patient data sets.csv"),
]

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lung_cancer_model.pkl")

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Air Pollution",
    "Alcohol use",
    "Dust Allergy",
    "OccuPational Hazards",
    "Genetic Risk",
    "chronic Lung Disease",
    "Balanced Diet",
    "Obesity",
    "Smoking",
    "Passive Smoker",
    "Chest Pain",
    "Coughing of Blood",
    "Fatigue",
    "Weight Loss",
    "Shortness of Breath",
    "Wheezing",
    "Swallowing Difficulty",
    "Clubbing of Finger Nails",
    "Frequent Cold",
    "Dry Cough",
    "Snoring",
]

LEVEL_MAPPING = {"Low": 0, "Medium": 1, "High": 2}


def _find_csv():
    for path in _CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _make_synthetic_data(n=1000, seed=42):
    """
    Generate a synthetic cancer-patient dataset that mirrors the structure of
    the original Kaggle dataset.  Each risk level gets predictable but noisy
    feature profiles so a Random Forest can learn meaningful boundaries.
    """
    rng = np.random.default_rng(seed)
    rows = []
    labels = []

    # Proportions roughly matching the original dataset
    per_level = {
        "Low":    int(n * 0.33),
        "Medium": int(n * 0.33),
        "High":   n - int(n * 0.33) - int(n * 0.33),
    }

    # Mean / std for each feature per risk level
    # (values fit within [1, 9] range used in the original)
    profile = {
        "Low": {
            "Age": (35, 8), "Gender": (1.5, 0.5), "Air Pollution": (2, 1),
            "Alcohol use": (3, 1), "Dust Allergy": (4, 1),
            "OccuPational Hazards": (3, 1), "Genetic Risk": (2, 1),
            "chronic Lung Disease": (2, 1), "Balanced Diet": (4, 1),
            "Obesity": (3, 1), "Smoking": (2, 1), "Passive Smoker": (2, 1),
            "Chest Pain": (2, 1), "Coughing of Blood": (2, 1), "Fatigue": (3, 1),
            "Weight Loss": (3, 1), "Shortness of Breath": (2, 1), "Wheezing": (2, 1),
            "Swallowing Difficulty": (2, 1), "Clubbing of Finger Nails": (2, 1),
            "Frequent Cold": (2, 1), "Dry Cough": (3, 1), "Snoring": (3, 1),
        },
        "Medium": {
            "Age": (45, 8), "Gender": (1.5, 0.5), "Air Pollution": (5, 1),
            "Alcohol use": (5, 1), "Dust Allergy": (5, 1),
            "OccuPational Hazards": (5, 1), "Genetic Risk": (4, 1),
            "chronic Lung Disease": (4, 1), "Balanced Diet": (5, 1),
            "Obesity": (5, 1), "Smoking": (5, 1), "Passive Smoker": (4, 1),
            "Chest Pain": (4, 1), "Coughing of Blood": (4, 1), "Fatigue": (5, 1),
            "Weight Loss": (4, 1), "Shortness of Breath": (4, 1), "Wheezing": (4, 1),
            "Swallowing Difficulty": (4, 1), "Clubbing of Finger Nails": (4, 1),
            "Frequent Cold": (4, 1), "Dry Cough": (5, 1), "Snoring": (4, 1),
        },
        "High": {
            "Age": (55, 8), "Gender": (1.5, 0.5), "Air Pollution": (7, 1),
            "Alcohol use": (7, 1), "Dust Allergy": (7, 1),
            "OccuPational Hazards": (7, 1), "Genetic Risk": (7, 1),
            "chronic Lung Disease": (7, 1), "Balanced Diet": (4, 1),
            "Obesity": (7, 1), "Smoking": (7, 1), "Passive Smoker": (7, 1),
            "Chest Pain": (7, 1), "Coughing of Blood": (7, 1), "Fatigue": (7, 1),
            "Weight Loss": (7, 1), "Shortness of Breath": (7, 1), "Wheezing": (7, 1),
            "Swallowing Difficulty": (7, 1), "Clubbing of Finger Nails": (7, 1),
            "Frequent Cold": (4, 1), "Dry Cough": (7, 1), "Snoring": (5, 1),
        },
    }

    for level, count in per_level.items():
        p = profile[level]
        for _ in range(count):
            row = {}
            for feat in FEATURE_COLUMNS:
                mu, sigma = p[feat]
                val = float(rng.normal(mu, sigma))
                # Clamp to valid range
                if feat == "Age":
                    val = int(np.clip(val, 15, 70))
                elif feat == "Gender":
                    val = int(np.clip(round(val), 1, 2))
                else:
                    val = int(np.clip(round(val), 1, 9))
                row[feat] = val
            rows.append(row)
            labels.append(LEVEL_MAPPING[level])

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df["Level"] = labels
    return df


def train():
    csv_path = _find_csv()
    if not csv_path:
        print("ERROR: Real dataset CSV not found.")
        print("Please run:  python download_data.py")
        print("This will download the dataset from Kaggle and retrain automatically.")
        raise SystemExit(1)

    print(f"Loading real dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.drop(columns=["index", "Patient Id"], inplace=True, errors="ignore")
    df["Level"] = df["Level"].map(LEVEL_MAPPING)

    missing = df["Level"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows had unrecognised Level values and will be dropped.")
        df = df.dropna(subset=["Level"])

    X = df[FEATURE_COLUMNS]
    y = df["Level"]

    print(f"Training RandomForestClassifier on {len(X)} real patient records ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {MODEL_PATH}")
    print("Done! Retrained on real data.")


if __name__ == "__main__":
    train()
