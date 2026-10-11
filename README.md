# Air-Image-Recognition-and-Prediction-Shape-Using-CNN

A Computer Vision and Deep Learning system that enables touchless mid-air drawing using hand gesture tracking, removes trajectory jitter via signal smoothing, classifies drawings using a 15-class Convolutional Neural Network (CNN), and post-processes strokes into vector graphics and rendered text.

---

## Table of Contents
- [Overview](#overview)
- [Research Background](#research-background)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [How It Works](#how-it-works)
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
