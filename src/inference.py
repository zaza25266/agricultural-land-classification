
"""Model loading and prediction for agricultural land classification."""

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from huggingface_hub import hf_hub_download

from src.preprocessing import (
    preprocess_for_keras,
    preprocess_for_pytorch,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HF_REPO_ID = "ZubairAli25266/agri-land-classification-models"

PYTORCH_FILENAME = "ai_capstone_pytorch_state_dict.pth"
KERAS_FILENAME = "ai_capstone_keras_best_model.model.keras"

CLASS_NAMES = {
    0: "Non-agricultural",
    1: "Agricultural",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# PyTorch architecture
# Checkpoint parameter indices match the saved state_dict.
# ------------------------------------------------------------


class ConvNet(nn.Sequential):
    def __init__(self, num_classes=2):
        super().__init__(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # 0
            nn.ReLU(),                                   # 1
            nn.MaxPool2d(2),                             # 2
            nn.BatchNorm2d(32),                          # 3

            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 4
            nn.ReLU(),                                   # 5
            nn.MaxPool2d(2),                             # 6
            nn.BatchNorm2d(64),                          # 7

            nn.Conv2d(64, 128, kernel_size=5, padding=2), # 8
            nn.ReLU(),                                   # 9
            nn.MaxPool2d(2),                              # 10
            nn.BatchNorm2d(128),                          # 11

            nn.Conv2d(128, 256, kernel_size=5, padding=2),# 12
            nn.ReLU(),                                    # 13
            nn.MaxPool2d(2),                              # 14
            nn.BatchNorm2d(256),                          # 15

            nn.Conv2d(256, 512, kernel_size=5, padding=2),# 16
            nn.ReLU(),                                    # 17
            nn.MaxPool2d(2),                              # 18
            nn.BatchNorm2d(512),                          # 19

            nn.Conv2d(512, 1024, kernel_size=5, padding=2),# 20
            nn.ReLU(),                                     # 21
            nn.MaxPool2d(2),                               # 22
            nn.BatchNorm2d(1024),                           # 23

            nn.AdaptiveAvgPool2d((1, 1)),                  # 24
            nn.Flatten(),                                  # 25
            nn.Linear(1024, 2048),                         # 26
            nn.ReLU(),                                     # 27
            nn.BatchNorm1d(2048),                          # 28
            nn.Dropout(0.4),                               # 29
            nn.Linear(2048, num_classes),                  # 30
        )

# ------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------

def load_pytorch_model():
    """Download and load the trained PyTorch checkpoint."""

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=PYTORCH_FILENAME,
    )

    model = ConvNet(num_classes=2)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


def load_keras_model():
    """Download and load the trained Keras model."""

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=KERAS_FILENAME,
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


# ------------------------------------------------------------
# Shared result formatter
# ------------------------------------------------------------

def format_result(probabilities, model_name):
    """Return prediction details in a consistent dictionary."""

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if len(probabilities) != 2:
        raise ValueError(
            f"Expected 2 class probabilities, got {len(probabilities)}."
        )

    predicted_class = int(np.argmax(probabilities))

    return {
        "model": model_name,
        "class_id": predicted_class,
        "prediction": CLASS_NAMES[predicted_class],
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(2)
        },
    }


# ------------------------------------------------------------
# PyTorch prediction
# ------------------------------------------------------------

def predict_pytorch(image, model=None):
    """Predict an image using PyTorch."""

    if model is None:
        model = load_pytorch_model()

    model.eval()

    processed = preprocess_for_pytorch(image)

    tensor = torch.as_tensor(
        processed,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    return format_result(
        probabilities.cpu().numpy(),
        "PyTorch",
    )


# ------------------------------------------------------------
# Keras prediction
# ------------------------------------------------------------

def predict_keras(image, model=None):
    """Predict an image using Keras."""

    if model is None:
        model = load_keras_model()

    processed = preprocess_for_keras(image)

    prediction = np.asarray(
        model.predict(processed, verbose=0)
    )

    # One sigmoid output: probability of the positive class.
    if prediction.size == 1:
        agricultural_probability = float(
            prediction.reshape(-1)[0]
        )

        probabilities = [
            1.0 - agricultural_probability,
            agricultural_probability,
        ]

    # Two softmax outputs: probabilities for both classes.
    elif prediction.shape[-1] == 2:
        probabilities = prediction.reshape(-1, 2)[0]

    else:
        raise ValueError(
            f"Unexpected Keras output shape: {prediction.shape}"
        )

    return format_result(
        probabilities,
        "TensorFlow / Keras",
    )
