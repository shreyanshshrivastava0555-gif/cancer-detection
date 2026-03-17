import os
import tensorflow as tf
from data_loader import get_data_generators
from model import build_model
import matplotlib.pyplot as plt

def train_model(train_dir, val_dir, epochs=10, batch_size=32, model_save_path='models/cancer_detection_model.h5'):
    """
    Trains the cancer detection model.
    """
    
    # 1. Get Data Generators
    print("Loading data...")
    train_gen, val_gen = get_data_generators(train_dir, val_dir, batch_size=batch_size)
    
    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    print(f"Detected {num_classes} classes: {class_indices}")

    # 2. Build Model
    print("Building model...")
    model = build_model(num_classes=num_classes)

    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')
    ]

    # 4. Train
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    # 5. Plotting Results
    plot_training_history(history)
    
    # 6. Save Labels
    labels_path = os.path.join(os.path.dirname(model_save_path), 'labels.txt')
    with open(labels_path, 'w') as f:
        for label, index in sorted(class_indices.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
    print(f"Labels saved to {labels_path}")
    
    return history, model

def plot_training_history(history):
    """
    Plots training and validation accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('models/training_history.png')
    print("Training plots saved to models/training_history.png")

if __name__ == "__main__":
    # Example paths - these should be updated with actual Kaggle dataset paths
    TRAIN_DIR = 'data/raw/train'
    VAL_DIR = 'data/raw/val'
    
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        train_model(TRAIN_DIR, VAL_DIR, epochs=20)
    else:
        print(f"Data directories not found at {TRAIN_DIR} and {VAL_DIR}.")
        print("Please download and organize your dataset first.")
