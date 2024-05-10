
---

# Handwritten Digit Recognition System

This project implements a system for capturing, processing, and recognizing handwritten digits using Python, OpenCV, and Keras. It includes functions for image capture, preprocessing, dataset creation, and model training.

## Features

- **Image Capture**: Captures images of handwritten digits through a webcam.
- **Image Preprocessing**: Processes images to the required size and format.
- **Dataset Splitting**: Splits images into training and test datasets.
- **Model Training**: Trains a convolutional neural network (CNN) on the dataset.

## Setup

To run this project, you need to have Python installed along with the following packages:
- OpenCV
- NumPy
- Keras
- scikit-learn

You can install these packages using pip:
```bash
pip install opencv-python numpy keras scikit-learn
```

## Usage

1. **Capture Images**: This script will use your default webcam to capture images of digits. Make sure your webcam is accessible and correctly configured on your system. Run the capturing function with:
   ```python
   capture_images()
   ```

2. **Preprocess Images**: To preprocess the captured images for model consumption, run:
   ```python
   preprocess_images("dataset", "preprocessed_dataset")
   ```

3. **Split Dataset**: Split the preprocessed images into training and testing datasets:
   ```python
   split_dataset("preprocessed_dataset", "train", "test")
   ```

4. **Train Model**: Train the CNN on the training dataset:
   ```python
   train_model("train", "test")
   ```

## File Descriptions

- `capture_images()`: Captures images of digits 0-9 using a webcam.
- `preprocess_images(input_dir, output_dir)`: Resizes images to 28x28 pixels and converts them to grayscale.
- `split_dataset(input_dir, train_dir, test_dir)`: Splits the dataset into training and testing sets.
- `train_model(train_dir, test_dir)`: Loads the dataset, creates the CNN model, and trains it.

## Model Architecture

The CNN model used includes the following layers:
- Conv2D: 32 filters, 3x3 kernel size, ReLU activation
- MaxPooling2D: 2x2 pool size
- Flatten
- Dense: 128 units, ReLU activation
- Dense: 10 units (output layer), softmax activation

The model is compiled with the Adam optimizer and categorical crossentropy loss function. It evaluates the accuracy of the trained model on the test dataset.

## Conclusion

This script provides a basic yet comprehensive approach to digit recognition from handwritten images. It demonstrates the end-to-end process from image capture to model training and evaluation.

---
