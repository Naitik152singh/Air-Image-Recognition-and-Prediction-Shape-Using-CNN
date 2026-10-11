import os
import urllib.request
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# --- 1. Class Labels Definition (15 Classes Total) ---
DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SHAPE_CLASSES = ['circle', 'triangle', 'square', 'star', 'line']
ALL_CLASSES = DIGIT_CLASSES + SHAPE_CLASSES

IMG_SIZE = 64
SAMPLES_PER_CLASS = 2500  # 2,500 samples per class

X_all, y_all = [], []

# --- 2. Load Handwritten Digits (MNIST) ---
print("[1/3] Loading MNIST Handwritten Digits (0-9)...")
(x_train_mnist, y_train_mnist), _ = mnist.load_data()

for digit_idx in range(10):
    # Filter images for specific digit
    digit_mask = (y_train_mnist == digit_idx)
    digit_imgs = x_train_mnist[digit_mask][:SAMPLES_PER_CLASS]
    
    # Resize 28x28 to 64x64
    resized_digit_imgs = [cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in digit_imgs]
    
    X_all.append(np.array(resized_digit_imgs))
    y_all.append(np.full(len(resized_digit_imgs), digit_idx))

# --- 3. Load Geometric Shapes (Google Quick, Draw!) ---
print("[2/3] Downloading & Loading Google Quick, Draw! Shapes...")
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DATA_DIR = "quickdraw_data"
os.makedirs(DATA_DIR, exist_ok=True)

for shape_idx, shape_name in enumerate(SHAPE_CLASSES):
    file_path = os.path.join(DATA_DIR, f"{shape_name}.npy")
    url = f"{BASE_URL}{shape_name}.npy"
    
    if not os.path.exists(file_path):
        print(f" -> Downloading {shape_name}.npy...")
        urllib.request.urlretrieve(url, file_path)
    
    raw_data = np.load(file_path)[:SAMPLES_PER_CLASS]
    shape_imgs = raw_data.reshape(-1, 28, 28)
    
    resized_shape_imgs = [cv2.resize(img, (IMG_SIZE, IMG_SIZE)) for img in shape_imgs]
    
    X_all.append(np.array(resized_shape_imgs))
    # Class index offsets by 10 (0-9 are digits, 10-14 are shapes)
    y_all.append(np.full(len(resized_shape_imgs), 10 + shape_idx))

# Combine, normalize, and reshape
X_train = np.concatenate(X_all, axis=0).astype('float32') / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (N, 64, 64, 1)
y_train = np.concatenate(y_all, axis=0)

print(f"\nUnified Dataset Assembled: {X_train.shape[0]} samples across {len(ALL_CLASSES)} classes.")

# --- 4. Build CNN Architecture ---
print("[3/3] Training 15-Class CNN Classifier...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Dropout(0.3),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(ALL_CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=12, batch_size=64, validation_split=0.2, shuffle=True)
model.save("shape_model.h5")
print("\n[SUCCESS] Model trained and saved as 'shape_model.h5'")