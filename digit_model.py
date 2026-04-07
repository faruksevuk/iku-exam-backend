"""
Simple MNIST CNN for digit recognition only.
Letters use TrOCR (more accurate for single handwritten letters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple
import os

_model = None
_model_path = os.path.join(os.path.dirname(__file__), 'models', 'digit_cnn.pt')


class MiniDigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*7*7, 64), nn.ReLU(), nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


def classify_digit(image: np.ndarray) -> Tuple[int, float]:
    """Classify digit 0-9."""
    global _model
    if _model is None:
        _model = MiniDigitCNN()
        _model.load_state_dict(torch.load(_model_path, map_location='cpu', weights_only=True))
        _model.eval()
        print("[DigitCNN] Loaded")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.mean() > 127:
        gray = 255 - gray

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.1307) / 0.3081

    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1)
        conf, pred = probs.max(1)
    return (pred.item(), round(conf.item(), 4))
