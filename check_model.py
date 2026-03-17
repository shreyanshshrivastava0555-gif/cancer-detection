import tensorflow as tf
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
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

print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("\nCommon fixes:")
    print("  • TF version mismatch  → reinstall TF matching the version used to train")
    print("  • Corrupt .h5 file     → re-export / re-download the model")
    print("  • Custom layers        → pass custom_objects= to load_model()")
    sys.exit(1)

# ── Architecture summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
model.summary()

# ── Shape info ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SHAPE INFO")
print("=" * 60)
print(f"  Input shape  : {model.input_shape}")
print(f"  Output shape : {model.output_shape}")
print(f"  Total params : {model.count_params():,}")

# ── Layer-level detail ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LAYER DETAILS")
print("=" * 60)
for i, layer in enumerate(model.layers):
    cfg = layer.get_config()
    activation = cfg.get("activation", "—")
    try:
        out_shape = layer.output_shape
    except AttributeError:
        out_shape = "N/A"
    trainable_params = sum(tf.size(w).numpy() for w in layer.trainable_weights)
    print(f"  [{i:>3}] {layer.name:<35} out={str(out_shape):<25} "
          f"act={str(activation):<12} trainable_params={trainable_params:,}")

# ── Output neuron hint ────────────────────────────────────────────────────────
n_classes = model.output_shape[-1]
print("\n" + "=" * 60)
print("DIAGNOSIS HINT")
print("=" * 60)
if n_classes == 1:
    print("  1 output neuron → binary classifier (benign vs malignant)")
    print("  Expected final activation: sigmoid")
else:
    print(f"  {n_classes} output neurons → multi-class classifier")
    print("  Expected final activation: softmax")
    print(f"  Make sure you have exactly {n_classes} class labels.")
