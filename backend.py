# pyre-ignore-all-errors
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import io
import os
import pickle

app = FastAPI(title="Cancer Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Skin Cancer Model (image-based)
# ---------------------------------------------------------------------------
SKIN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skin_cancer_model.h5')
try:
    print(f"Loading skin cancer model from {SKIN_MODEL_PATH}...")
    def dummy_dense(**kwargs):
        kwargs.pop('quantization_config', None)
        return tf.keras.layers.Dense(**kwargs)
    skin_model = tf.keras.models.load_model(SKIN_MODEL_PATH, custom_objects={'Dense': dummy_dense}, compile=False)
    print("Skin cancer model loaded successfully.")
except Exception as e:
    print(f"Error loading skin cancer model: {e}")
    skin_model = None

SKIN_CLASS_NAMES = [
    'Actinic Keratosis',       # akiec (0)
    'Basal Cell Carcinoma',     # bcc   (1)
    'Benign Keratosis',         # bkl   (2)
    'Dermatofibroma',           # df    (3)
    'Melanoma',                 # mel   (4)
    'Melanocytic Nevi',         # nv    (5)
    'Vascular Lesion'           # vasc  (6)
]
IMG_SIZE = 64

# ---------------------------------------------------------------------------
# Lung Cancer Model (tabular / Random Forest)
# ---------------------------------------------------------------------------
LUNG_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'lung_cancer_model.pkl')
try:
    print(f"Loading lung cancer model from {LUNG_MODEL_PATH}...")
    with open(LUNG_MODEL_PATH, 'rb') as f:
        lung_model = pickle.load(f)
    print("Lung cancer model loaded successfully.")
except Exception as e:
    print(f"Error loading lung cancer model: {e}")
    lung_model = None

LUNG_FEATURE_COLUMNS = [
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
LUNG_LEVEL_LABELS = {0: "Low", 1: "Medium", 2: "High"}


class LungCancerInput(BaseModel):
    Age: int
    Gender: int
    Air_Pollution: int
    Alcohol_use: int
    Dust_Allergy: int
    OccuPational_Hazards: int
    Genetic_Risk: int
    chronic_Lung_Disease: int
    Balanced_Diet: int
    Obesity: int
    Smoking: int
    Passive_Smoker: int
    Chest_Pain: int
    Coughing_of_Blood: int
    Fatigue: int
    Weight_Loss: int
    Shortness_of_Breath: int
    Wheezing: int
    Swallowing_Difficulty: int
    Clubbing_of_Finger_Nails: int
    Frequent_Cold: int
    Dry_Cough: int
    Snoring: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict_skin(file: UploadFile = File(...)):
    if skin_model is None:
        raise HTTPException(status_code=500, detail="Skin cancer model failed to load.")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = skin_model.predict(img_array)
        pred_probs = prediction[0].tolist()
        pred_idx = int(np.argmax(pred_probs))
        confidence = float(pred_probs[pred_idx])
        
        probabilities = [
            {"class": SKIN_CLASS_NAMES[i], "probability": prob}
            for i, prob in enumerate(pred_probs)
        ]
        probabilities.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "prediction": SKIN_CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "probabilities": probabilities
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


@app.post("/predict-lung-cancer")
async def predict_lung(data: LungCancerInput):
    if lung_model is None:
        raise HTTPException(status_code=500, detail="Lung cancer model failed to load. Run train_lung_model.py first.")
    
    try:
        # Build feature dataframe in the exact format the model expects
        features = pd.DataFrame([{
            "Age": data.Age,
            "Gender": data.Gender,
            "Air Pollution": data.Air_Pollution,
            "Alcohol use": data.Alcohol_use,
            "Dust Allergy": data.Dust_Allergy,
            "OccuPational Hazards": data.OccuPational_Hazards,
            "Genetic Risk": data.Genetic_Risk,
            "chronic Lung Disease": data.chronic_Lung_Disease,
            "Balanced Diet": data.Balanced_Diet,
            "Obesity": data.Obesity,
            "Smoking": data.Smoking,
            "Passive Smoker": data.Passive_Smoker,
            "Chest Pain": data.Chest_Pain,
            "Coughing of Blood": data.Coughing_of_Blood,
            "Fatigue": data.Fatigue,
            "Weight Loss": data.Weight_Loss,
            "Shortness of Breath": data.Shortness_of_Breath,
            "Wheezing": data.Wheezing,
            "Swallowing Difficulty": data.Swallowing_Difficulty,
            "Clubbing of Finger Nails": data.Clubbing_of_Finger_Nails,
            "Frequent Cold": data.Frequent_Cold,
            "Dry Cough": data.Dry_Cough,
            "Snoring": data.Snoring,
        }])
        
        prediction = lung_model.predict(features)[0]
        proba = lung_model.predict_proba(features)[0].tolist()
        level_label = LUNG_LEVEL_LABELS.get(int(prediction), "Unknown")
        confidence = float(max(proba))
        
        return {
            "level": level_label,
            "level_numeric": int(prediction),
            "confidence": confidence,
            "probabilities": {
                "Low": proba[0],
                "Medium": proba[1],
                "High": proba[2],
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process lung cancer data: {str(e)}")


# Mount the frontend directory to serve on / (must be at the bottom)
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
