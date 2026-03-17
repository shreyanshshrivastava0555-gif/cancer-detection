# Cancer Detection

AI-Powered Early Detection & Diagnostic System. A deep learning-based system for detecting multiple types of cancer using transfer learning and Streamlit.

## Features
- Detects Lung and Skin cancer (Skin lesion classification using HAM10000).
- Uses Transfer Learning and Random Forest models.
- Interactive Streamlit dashboard for real-time inference.
- Git LFS support for large model files.

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
