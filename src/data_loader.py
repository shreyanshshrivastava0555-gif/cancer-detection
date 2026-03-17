import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    """
    Creates training and validation data generators with augmentation.
    
    Args:
        train_dir: Path to training data directory.
        val_dir: Path to validation data directory.
        img_size: Target size for images.
        batch_size: Batch size for training.
        
    Returns:
        train_generator, val_generator
    """
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator

if __name__ == "__main__":
    # Example usage / testing script
    print("TensorFlow version:", tf.__version__)
    # Note: Directories need to exist for flow_from_directory to work
    # This is a placeholder for development
    print("Data loader module initialized.")
