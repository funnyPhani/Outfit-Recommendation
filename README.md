

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


