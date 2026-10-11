# Air-Image-Recognition-and-Prediction-Shape-Using-CNN

A Computer Vision and Deep Learning system that enables touchless mid-air drawing using hand gesture tracking, removes trajectory jitter via signal smoothing, classifies drawings using a 15-class Convolutional Neural Network (CNN), and post-processes strokes into vector graphics and rendered text.

---

## Table of Contents
- [Overview](#overview)
- [Research Background](#research-background)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Dataset Details](#dataset-details)
- [Gesture Controls & Keyboard Shortcuts](#gesture-controls--keyboard-shortcuts)
- [Troubleshooting](#troubleshooting)
- [Citation & References](#citation--references)

---

## Overview

Air-drawing interfaces allow users to sketch in 3D space using webcam feeds without touchscreens or physical styluses. However, freehand air-drawing suffers from **hand tremors, trajectory jitter, scale/orientation variances, and real-time inference latency**. 

This project implements an automated, single-script pipeline (`main.py`) that addresses these issues. It tracks finger movements in real time using MediaPipe, smooths coordinate signals using an Exponential Moving Average (EMA) filter, classifies strokes across 15 classes using a CNN trained on MNIST and Google Quick, Draw! data, and snaps wobbly hand-drawn strokes into clean vector geometry or rendered typography.

---

## Research Background

Inspired by recent research in spatial character and gesture recognition (*Zhang et al., Sensors 2023*), this system directly addresses four paper-defined challenges:

| Research Problem | Cause | Implemented Solution |
| :--- | :--- | :--- |
| **Air-Drawing Jitter** | Absence of tactile feedback causes erratic coordinate tracks. | **EMA Trajectory Filter:** Dampens high-frequency movement variations. |
| **Scale & Aspect Distortion** | Drawings vary in size and aspect ratio depending on distance to camera. | **1:1 Square Padding & Rescaling:** Crops bounding box and centers tensor to $64 \times 64$. |
| **Real-Time Latency** | Heavy deep learning models introduce webcam frame lag. | **Lightweight CNN Architecture:** Uses low-depth convolutions and MaxPool layers for sub-10ms inference. |
| **Imperfect Visual Quality** | Raw freehand lines remain messy even when correctly classified. | **Dynamic Vector Snapping:** Applies `cv2.approxPolyDP` and enclosing bounding functions for shape rendering. |

---

## Key Features

* **Single-Script Automated Setup:** `main.py` automatically checks for model weights, downloads required training datasets on first launch, trains the 15-class CNN, and starts the live app.
* **Jitter Reduction:** Real-time trajectory filtering produces smooth drawing strokes on screen.
* **15-Class Dual Classifier:**
  * **Digits ($0\text{--}9$):** Trained on the **MNIST** dataset.
  * **Geometric Shapes:** Trained on **Google Quick, Draw!** (`circle`, `triangle`, `square`, `star`, `line`).
* **Vector Post-Processing:** Converts messy hand-drawn shapes into clean geometry (`cv2.approxPolyDP`, `cv2.minEnclosingCircle`, `cv2.fitLine`) and digits into digital text typography.

---

## System Architecture

[ Webcam Feed ]
│
▼
[ MediaPipe Hand Tracking ] ──► Extract Index Fingertip Coordinates (Landmark 8)
│
▼
[ Signal Filtering ] ──► Exponential Moving Average (EMA) Smoothing
│
▼
[ Virtual Canvas Buffer ] ──► Accumulate Freehand Stroke Pixels
│
▼
[ Bounding Box Crop & Normalization ] ──► Pad 1:1 Aspect Ratio & Resize to 64x64
│
▼
[ 15-Class CNN Classifier ] ──► Compute Softmax Probability
│
├─────────────────────────────────┐
▼                                 ▼
[ Digit Handler (0-9) ]         [ Shape Vector Snapper ]
Render Clean Font Text          Polygonal / Circular Contour Fitting

---

## Tech Stack

* **Programming Language:** Python 3.8+
* **Computer Vision:** OpenCV (`opencv-python`)
* **Hand Tracking & Landmarks:** Google MediaPipe (`mediapipe`)
* **Deep Learning Framework:** TensorFlow / Keras (`tensorflow`)
* **Data Processing:** Scientific NumPy (`numpy`)

---

## Dataset Details

Category Type,Classes Included,Source Dataset,Image Dimensions
Handwritten Digits,"0, 1, 2, 3, 4, 5, 6, 7, 8, 9",MNIST Database,28×28 resized to 64×64
Geometric Shapes,"circle, triangle, square, star, line","Google Quick, Draw! Dataset",28×28 resized to 64×64

---

## Gesture Controls & Keyboard Shortcuts

Control Action,Gesture / Key Input,Description
Draw Mode,"Index Finger UP, Middle Finger DOWN",Draws green strokes onto the screen canvas.
Move Cursor,"Index Finger UP, Middle Finger UP",Repositions the active cursor without drawing.
Predict & Snap,Press p,Runs CNN inference on canvas. Replaces strokes with text or vector shapes.
Clear Canvas,Press c,Erases all drawn elements from the screen.
Quit App,Press q,Closes the application window and exits the process.

---

## Troubleshooting

Webcam Initialization Error (cap.isOpened() == False):
Change cap = cv2.VideoCapture(0) to cap = cv2.VideoCapture(1) inside main.py if using an external USB webcam.

Jitter Sensitivity:
Adjust the smoothing weight alpha in apply_ema_filter(..., alpha=0.35) inside main.py. A lower alpha (e.g., 0.20) increases stroke smoothness.

Low Accuracy on Complex Shapes:
Increase SAMPLES_PER_CLASS from 2500 to 5000 or increase EPOCHS in train_and_save_model() before running initial training.

## Citation & References

@article{zhang2023airgr,
  title={Air-GR: An Over-the-Air Handwritten Character Recognition System Based on Coordinate Correction YOLOv5 Algorithm and LGR-CNN},
  author={Zhang, Yajun and Li, Zijian and Yang, Zhixiong and Yuan, Bo and Liu, Xu},
  journal={Sensors},
  volume={23},
  number={3},
  pages={1464},
  year={2023},
  publisher={MDPI},
  doi={10.3390/s23031464}
}
