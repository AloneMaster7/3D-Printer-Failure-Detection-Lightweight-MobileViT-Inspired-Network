# 3D-Printer-Failure-Detection-Lightweight-MobileViT-Inspired-Network

A lightweight deep learning project for binary classification of 3D printer images, focusing on detecting **under-extrusion defects**.

## Model

The project implements a compact CNN inspired by lightweight MobileViT-style models, using PyTorch. The network consists of three convolutional blocks (16, 32, and 64 channels), followed by adaptive average pooling and a fully connected classifier.

> The implemented architecture is a simplified CNN, not the full MobileViT transformer architecture.

## Data Augmentation

A custom image augmentation function supports:
- Random rotation, translation, and scaling
- Gaussian noise injection
- Lightweight Gaussian blur

## Evaluation

To reduce potential data leakage between frames from the same video, **5-fold GroupKFold cross-validation** is performed using video IDs extracted from image filenames.

**Mean final validation accuracy: 99.17% ± 0.37%** across the five folds.

## Inference

The trained model can generate predictions for test images and export them to a CSV file compatible with the provided submission format.

## Technologies

Python · PyTorch · OpenCV · NumPy · Scikit-learn · Matplotlib
