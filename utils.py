import numpy as np
import random

def categorize_face(face_image):
    """
    Categorizes a face image as casual or formal based on its color distribution.
    Args:
        face_image (np.array): Input face image.
    Returns:
        int: 0 for casual, 1 for formal.
    """
    avg_color = np.mean(face_image, axis=(0, 1))
    return 0 if avg_color[1] > avg_color[0] else 1

def recommend_outfit(face_category):
    style_map = {
        0: ['T-shirt/top', 'Pullover', 'Sneaker', 'Shirt', 'Jeans', 'Hoodie'],
        1: ['Dress', 'Coat', 'Blazer', 'Suit', 'Tie', 'Formal shoes']
    }
    outfits = style_map.get(face_category, [])
    return random.choice(outfits) if outfits else 'No recommendation available'
