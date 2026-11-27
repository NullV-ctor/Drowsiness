"""Shared preprocessing and portable linear model used by training and inference."""

from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = "equalized-hog-64x32-v1"
IMAGE_SIZE = (64, 32)  # width, height


def prepare_eye(image):
    if image is None or image.size == 0:
        raise ValueError("Empty eye image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray.astype(np.uint8))


def extract_features(image):
    # Create per call: no mutable HOG descriptor shared between camera threads.
    hog = cv2.HOGDescriptor(IMAGE_SIZE, (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(prepare_eye(image)).ravel()


class EyeClassifier:
    """NumPy inference for a notebook-trained StandardScaler + LinearSVC."""

    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["feature_version"].item()) != FEATURE_VERSION:
                raise ValueError(
                    "Incompatible model. Retrain with the current notebook."
                )
            self.mean = data["mean"].copy()
            self.scale = data["scale"].copy()
            self.coef = data["coef"].ravel().copy()
            self.intercept = float(data["intercept"].item())
        size = extract_features(np.zeros((32, 64), np.uint8)).size
        if any(a.shape != (size,) for a in (self.mean, self.scale, self.coef)):
            raise ValueError("Invalid model feature dimensions.")
        if not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.coef)):
            raise ValueError("Invalid model values.")
        if not np.isfinite(self.intercept) or np.any(self.scale <= 0):
            raise ValueError("Invalid model scaling.")

    def decision_function(self, images):
        features = np.stack([extract_features(image) for image in images])
        return ((features - self.mean) / self.scale) @ self.coef + self.intercept

    def predict(self, images):
        # Training labels: open=0, closed=1. A margin is NOT a probability.
        return (self.decision_function(images) >= 0).astype(int)
