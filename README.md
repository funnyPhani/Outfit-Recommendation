

# Face-Based Outfit Recommendation System

This repository implements a **Face-Based Outfit Recommendation System** using a combination of **CelebA** images for face classification and **Fashion MNIST** for outfit suggestions. The project leverages transfer learning with a VGG16-based model.

---

## Features
- **Image Preprocessing:** Handles resizing, normalization, and RGB channel addition for datasets.
- **Face Classification:** Categorizes face images into *Casual* or *Formal* based on trained predictions.
- **Outfit Recommendation:** Suggests outfits from Fashion MNIST categories based on face classification results.
- **Transfer Learning:** Utilizes VGG16 pre-trained on ImageNet for feature extraction.
- **Dynamic Data Loading:** Includes a generator for processing large-scale CelebA data.
- **Modular Design:** Organized into separate components for easy maintainability.

---

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/face-outfit-recommendation.git
   cd face-outfit-recommendation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the datasets**:
   - [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Save images in `archive/img_align_celeba/`.
   - [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) - Automatically handled by the script.

4. **Enable mixed precision (optional)**:
   - Mixed precision is supported via TensorFlow's `mixed_float16` policy. Make sure your hardware supports it (e.g., NVIDIA GPUs with Tensor Cores).

---

## Project Structure

```plaintext
face-outfit-recommendation/
├── data_preprocessing.py      # Handles data loading and preprocessing
├── model_builder.py           # Defines the transfer learning model
├── utils.py                   # Contains helper functions for categorization and recommendations
├── main.py                    # Main script for training, validation, and inference
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

## Usage

### Train the Model
Run the following command to preprocess data, build the model, and train it:
```bash
python main.py
```

### Test the Model
Update the `sample_image` path in `main.py` to point to a test image from the CelebA dataset. Then, run:
```bash
python main.py
```

The system will:
1. Predict whether the input face is *Casual* or *Formal*.
2. Recommend an outfit from the Fashion MNIST categories.
3. Display the input image and the recommended outfit.

---

## Example Output
Input image and outfit recommendation will appear as a side-by-side plot:
1. **Input Face Image:** The selected CelebA image.
2. **Recommended Outfit:** An item from Fashion MNIST.

---

## Dependencies
- Python 3.7+
- TensorFlow 2.5+
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## Dataset Details

### CelebA
- **Use:** Face classification into casual/formal categories.
- **Path:** `archive/img_align_celeba/`

### Fashion MNIST
- **Use:** Outfit category suggestions.
- **Categories Used:** `T-shirt/top`, `Pullover`, `Dress`, `Coat`, `Sandal`, and others.

---

## Acknowledgments
- [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) by MMLab.
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research.
- [VGG16 Model](https://keras.io/api/applications/vgg/#vgg16-function).

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Sample output
![Sample Output](https://github.com/funnyPhani/Outfit-Recommendation/blob/main/Screenshot%202024-11-27%20123739.png)
![Sample Output](https://github.com/funnyPhani/Outfit-Recommendation/blob/main/Screenshot%202024-11-27%20123647.png)
![Input Face Image](https://github.com/funnyPhani/Outfit-Recommendation/blob/main/Screenshot%202024-11-27%20125850.png)
![Recommended Outfit](https://github.com/funnyPhani/Outfit-Recommendation/blob/main/Screenshot%202024-11-27%20125935.png)

---


Here’s a detailed documentation on how the **Face-Based Outfit Recommendation System** pipeline works:

---

## **Face-Based Outfit Recommendation System**
This project aims to recommend outfits based on facial characteristics, leveraging deep learning and transfer learning techniques.

### **Pipeline Overview**
1. **Data Preprocessing**
2. **Model Training**
3. **Inference and Recommendation**
4. **Visualization**

---

### **1. Data Preprocessing**

#### **a. CelebA Dataset Preprocessing**
- **Input:** CelebA dataset containing facial images.
- **Steps:**
  1. Images are resized to a standard size (128x128) for uniformity using OpenCV.
  2. Normalized to the range [0, 1].
  3. Labels are generated based on a simple rule (categorizing images as *Casual* or *Formal* based on average color intensity).
  
#### **b. Fashion MNIST Dataset Preprocessing**
- **Input:** Fashion MNIST dataset with grayscale clothing images.
- **Steps:**
  1. Images are resized to 128x128 and converted to RGB format by duplicating channels.
  2. A subset of images is selected (6 categories relevant to the recommendation system).
  3. Images are normalized to [0, 1].

---

### **2. Model Training**

#### **Transfer Learning with VGG16**
- **Architecture:**
  - **Base Model:** Pre-trained VGG16 (ImageNet weights) is used as the feature extractor.
  - **Head Layers:**
    - Global Average Pooling.
    - Dense layers with dropout for regularization.
    - Output layer with a sigmoid activation for binary classification (Casual or Formal).
- **Training:**
  - **Training Data:** CelebA data generator is used to feed preprocessed batches.
  - **Validation Data:** Fashion MNIST subset (simulates clothing categories for testing).
  - **Callbacks:** 
    - Early stopping.
    - Learning rate reduction on plateau.
    - Model checkpointing.
- **Mixed Precision Training:** Enabled to reduce memory usage and improve training efficiency.

---

### **3. Inference and Recommendation**

#### **Steps:**
1. **Face Image Preprocessing:**
   - Input face image is resized, normalized, and passed to the trained model.
2. **Category Prediction:**
   - The model predicts whether the face corresponds to *Casual* or *Formal*.
3. **Outfit Recommendation:**
   - Based on the predicted category, an outfit is randomly chosen from predefined clothing styles:
     - **Casual:** T-shirt, Pullover, Jeans, Hoodie, etc.
     - **Formal:** Dress, Blazer, Suit, Tie, etc.

---

### **4. Visualization**

#### **Steps:**
1. The input face image and the recommended outfit image are displayed side by side using Matplotlib.
2. If no matching outfit is found in the Fashion MNIST dataset for the recommended style, an appropriate message is displayed.

---

### Model Training

![Sample Output](https://github.com/funnyPhani/Outfit-Recommendation/blob/main/Screenshot%202024-11-27%20125247.png)




