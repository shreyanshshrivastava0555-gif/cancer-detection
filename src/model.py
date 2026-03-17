import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras import layers, models

def build_model(model_type='resnet50', num_classes=3, input_shape=(224, 224, 3)):
    """
    Builds a transfer learning model based on ResNet50 or MobileNetV2.
    
    Args:
        model_type: 'resnet50' or 'mobilenetv2'.
        num_classes: Number of output classes (cancer types).
        input_shape: Input image shape.
        
    Returns:
        Compiled Keras model.
    """
    
    if model_type.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_type.lower() == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model type. Choose 'resnet50' or 'mobilenetv2'.")

    # Freeze the base model layers
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Metrics: Accuracy, Precision, Recall
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        # F1 Score is not built-in for multi-class in basic Keras, 
        # but we can monitor it or calculate it during evaluation.
    ]

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=metrics
    )

    return model

if __name__ == "__main__":
    # Test building the model
    model = build_model(num_classes=4)
    model.summary()
    print("Model built and compiled successfully.")
