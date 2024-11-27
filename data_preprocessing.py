import os
import cv2
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def preprocess_celebA_image(image_path, target_size=(128, 128)):
    """
    Preprocesses a single image from the CelebA dataset.
    Args:
        image_path (str): Path to the image.
        target_size (tuple): Target dimensions for resizing.
    Returns:
        np.array: Normalized and resized image.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image / 255.0

def celebA_data_generator(celebA_images_folder, batch_size, target_size=(128, 128)):
    """
    Generator to load CelebA data in batches.
    Args:
        celebA_images_folder (str): Folder containing CelebA images.
        batch_size (int): Number of images per batch.
    Yields:
        np.array, np.array: Batch of images and corresponding labels.
    """
    image_files = [f for f in os.listdir(celebA_images_folder) if f.endswith('.jpg')]
    while True:
        np.random.shuffle(image_files)
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            images, labels = [], []
            for f in batch_files:
                img = preprocess_celebA_image(os.path.join(celebA_images_folder, f), target_size)
                label = categorize_face(img)
                images.append(img)
                labels.append(label)
            yield np.array(images), np.array(labels)

def preprocess_fashion_mnist_images(images, target_size=(128, 128)):
        """
    Resize Fashion MNIST images to match the target size and add RGB channels.
    Args:
        images (np.array): Array of grayscale images.
        target_size (tuple): Target dimensions for resizing.
    Returns:
        np.array: Resized images with 3 channels.
    """
    resized_images = [cv2.resize(img, target_size) for img in images]
    return np.array([np.stack([img] * 3, axis=-1) / 255.0 for img in resized_images])

def load_limited_fashion_mnist(n_images_per_class=5000):
        """
    Loads a limited subset of Fashion MNIST dataset, restricting to n_images_per_class images per class.
    Returns:
        tuple: Arrays of training and test data (X_train, y_train, X_test, y_test).
    """
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    selected_images, selected_labels = [], []
    for i in range(6):  # Classes 0-5
        indices = np.where(y_test == i)[0][:n_images_per_class]
        selected_images.extend(X_test[indices])
        selected_labels.extend([i] * len(indices))
    return preprocess_fashion_mnist_images(selected_images), np.array(selected_labels)
