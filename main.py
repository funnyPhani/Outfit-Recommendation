import logging
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_preprocessing import load_limited_fashion_mnist, celebA_data_generator
from model_builder import build_transfer_learning_model
from utils import recommend_outfit, preprocess_image_for_inference
from matplotlib import pyplot as plt
import cv2

# Logging configuration
logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler("training_log.log")
])

# Enable mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Paths
celebA_images_folder = 'archive/img_align_celeba/img_align_celeba'

# Data
X_fashion_train, y_fashion_train = load_limited_fashion_mnist()
celebA_generator = celebA_data_generator(celebA_images_folder, batch_size=8)

X_train_fashion, X_val_fashion, y_train_fashion, y_val_fashion = train_test_split(
    X_fashion_train, y_fashion_train, test_size=0.2, random_state=42)

# Model
model = build_transfer_learning_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
]

# Training
history = model.fit(
    celebA_generator,
    steps_per_epoch=200,
    epochs=20,
    validation_data=(X_val_fashion, y_val_fashion),
    callbacks=callbacks
)

# Save model
model.save('final_model.h5')
logging.info("Final model saved.")

# Inference and recommendation
def predict_and_recommend_outfit(image_path, model):
    # Preprocess the face image for inference
    face_image = preprocess_image_for_inference(image_path)
    prediction = model.predict(face_image)
    face_category = int(prediction[0][0] > 0.5)  # 0: Casual, 1: Formal

    # Recommend an outfit based on the face category
    recommended_outfit = recommend_outfit(face_category)
    print(f"Predicted Face Category: {'Casual' if face_category == 0 else 'Formal'}")
    print(f"Recommended Outfit: {recommended_outfit}")

    # Load and prepare the input face image
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Retrieve the Fashion MNIST image matching the recommendation
    fashion_mnist_image = get_fashion_mnist_image(recommended_outfit)
    if fashion_mnist_image is not None:
        fashion_mnist_image = cv2.resize(fashion_mnist_image, (128, 128))  # Resize for uniform display
        fashion_mnist_image = cv2.cvtColor(fashion_mnist_image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    else:
        print("No matching Fashion MNIST image found for the recommended outfit.")
        return

    # Plot the images side by side with titles
    plt.figure(figsize=(10, 5))
    
    # Display the input image
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Input Face Image")
    plt.axis('off')

    # Display the recommended outfit image
    plt.subplot(1, 2, 2)
    plt.imshow(fashion_mnist_image)
    plt.title(f"Recommended Outfit: {recommended_outfit}")
    plt.axis('off')

    # Show the combined plot
    plt.tight_layout()
    plt.show()

sample_image = 'archive/img_align_celeba/img_align_celeba/000545.jpg'
predict_and_display(sample_image)
