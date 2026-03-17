import tensorflow as tf
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
# Change this to your model filename; script looks next to itself first,
# then falls back to an absolute path if MODEL_PATH env var is set.
MODEL_FILENAME = "skin_cancer_model.h5"
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME),
)
# ─────────────────────────────────────────────────────────────────────────────

print(f"TensorFlow version : {tf.__version__}")
print(f"Python version     : {sys.version}")
print(f"Looking for model  : {MODEL_PATH}\n")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at: {MODEL_PATH}")
    print("Tip: set the MODEL_PATH environment variable or place the .h5 file "
          "in the same directory as this script.")
    sys.exit(1)

print(f"Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!\n")

    print(f"  Input shape  : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")
    print(f"  Total params : {model.count_params():,}")

    # Warn if output neurons look off for binary vs multi-class skin cancer
    n_classes = model.output_shape[-1]
    if n_classes == 1:
        print("\n  Output: 1 neuron → binary classification (sigmoid expected)")
    else:
        print(f"\n  Output: {n_classes} neurons → multi-class classification "
              "(softmax expected)")

except Exception as e:
    print(f"ERROR loading model: {e}")
    print("\nCommon fixes:")
    print("  • TF version mismatch  → reinstall TF matching the version used to train")
    print("  • Corrupt .h5 file     → re-export / re-download the model")
    print("  • Custom layers        → pass custom_objects= to load_model()")
    sys.exit(1)
