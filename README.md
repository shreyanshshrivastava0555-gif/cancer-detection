# Multi-Cancer Detection System

A deep learning-based system for detecting multiple types of cancer using transfer learning and Streamlit.

## Features
- Detects Breast, Lung, Skin, and other cancer types.
- Uses Transfer Learning with ResNet50/MobileNetV2.
- Interactive Streamlit dashboard for real-time inference.
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score).

## Project Structure
```
.
├── app.py              # Streamlit Application
├── requirements.txt    # Python dependencies
├── src/
│   ├── data_loader.py  # Image preprocessing and augmentation
│   ├── model.py        # Model architecture
│   └── train.py        # Training script
├── data/               # Dataset directory
└── models/             # Saved model weights
```

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training
Run the training script:
```bash
python src/train.py
```

### Dashboard
Start the Streamlit app:
```bash
streamlit run app.py
```
