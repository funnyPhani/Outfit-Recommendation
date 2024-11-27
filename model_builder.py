from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_transfer_learning_model(input_shape=(128, 128, 3)):
    """
    Builds a transfer learning model using VGG16 as the base model.
    Args:
        input_shape (tuple): Shape of the input data.
    Returns:
        tf.keras.Model: Compiled transfer learning model.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
