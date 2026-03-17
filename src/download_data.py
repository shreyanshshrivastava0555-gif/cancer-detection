import os
import requests
import zipfile
import shutil
import time

def download_file(url, target_path, max_retries=10, delay=30):
    print(f"Downloading {url} to {target_path}...")
    for attempt in range(max_retries):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {target_path}")
            return True
        elif response.status_code == 202:
            print(f"Server returned 202 (Accepted). File is being prepared. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return False
    return False

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def organize_dataset():
    # Figshare Direct Download Links
    datasets = {
        "Lung": "https://figshare.com/ndownloader/files/52627007", # Lung Cancer Detection - Dataset.zip (Wait for 202)
        "Breast": "https://figshare.com/ndownloader/files/36417579", # Updated Breast Cancer link
        "Skin": "https://figshare.com/ndownloader/files/40011718"   # Skin Cancer link
    }
    
    raw_dir = "data/raw"
    temp_dir = "data/temp"
    
    # Ensure clean state
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    for name, url in datasets.items():
        zip_path = os.path.join(temp_dir, f"{name}.zip")
        extract_path = os.path.join(temp_dir, name)
        
        if download_file(url, zip_path):
            if extract_zip(zip_path, extract_path):
                target_train = os.path.join(raw_dir, "train", name)
                target_val = os.path.join(raw_dir, "val", name)
                
                os.makedirs(target_train, exist_ok=True)
                os.makedirs(target_val, exist_ok=True)
                
                # Find image files
                files = []
                for root, _, filenames in os.walk(extract_path):
                    for f in filenames:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            files.append(os.path.join(root, f))
                
                if not files:
                    print(f"No images found for {name}")
                    continue
                    
                split_idx = int(len(files) * 0.8)
                train_files = files[:split_idx]
                val_files = files[split_idx:]
                
                print(f"Moving {len(train_files)} to train/{name} and {len(val_files)} to val/{name}...")
                
                for i, f in enumerate(train_files):
                    shutil.move(f, os.path.join(target_train, f"{name}_train_{i}.jpg"))
                for i, f in enumerate(val_files):
                    shutil.move(f, os.path.join(target_val, f"{name}_val_{i}.jpg"))
            else:
                print(f"Skipping {name} due to extraction failure.")
        else:
            print(f"Skipping {name} due to download failure.")
            
    # Cleanup temp
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print("Data acquisition complete.")

if __name__ == "__main__":
    organize_dataset()
