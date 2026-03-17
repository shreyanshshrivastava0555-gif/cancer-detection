"""
One-time script to download the real lung cancer dataset from Kaggle.

To get your Kaggle API key:
  1. Go to https://www.kaggle.com → Your Profile → Settings → API → Create New Token
  2. This downloads kaggle.json with {"username":...,"key":...}
  3. Enter those values when this script prompts you.
"""
import os
import json
import zipfile
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_DIR = os.path.expanduser("~/.config/kaggle")
KAGGLE_JSON = os.path.join(KAGGLE_DIR, "kaggle.json")

DATASET = "thedevastator/cancer-patients-and-air-pollution-a-new-link"

def setup_credentials():
    if os.path.exists(KAGGLE_JSON):
        print(f"Found existing Kaggle credentials at {KAGGLE_JSON}")
        return True
    print("\nKaggle API credentials needed.")
    print("Get them at: https://www.kaggle.com → Settings → API → Create New Token\n")
    username = input("Enter your Kaggle username: ").strip()
    key = input("Enter your Kaggle API key: ").strip()
    if not username or not key:
        print("ERROR: Username and key cannot be empty.")
        return False
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    with open(KAGGLE_JSON, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(KAGGLE_JSON, 0o600)
    print(f"Credentials saved to {KAGGLE_JSON}")
    return True

def download():
    if not setup_credentials():
        return
    print(f"\nDownloading dataset: {DATASET}")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET,
         "--unzip", "-p", BASE_DIR],
        capture_output=False
    )
    if result.returncode != 0:
        print("ERROR: Download failed. Check your credentials or internet connection.")
        return

    # Find the CSV
    for root, dirs, files in os.walk(BASE_DIR):
        for fname in files:
            if fname.endswith(".csv") and "cancer" in fname.lower():
                full = os.path.join(root, fname)
                print(f"\nDataset CSV found at: {full}")
                print("\nNow training the model with real data...")
                os.system(f'"{sys.executable}" "{os.path.join(BASE_DIR, "train_lung_model.py")}"')
                return

    print("CSV not found after download. Check the dataset folder.")

if __name__ == "__main__":
    download()
