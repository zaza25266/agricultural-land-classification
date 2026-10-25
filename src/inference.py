
"""Load pretrained models and run agricultural-land classification."""

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from huggingface_hub import hf_hub_download

from src.preprocessing import (
    preprocess_for_keras,
    preprocess_for_pytorch,
)


HF_REPO_ID = "ZubairAli25266/agri-land-classification-models"

KERAS_FILENAME = "ai_capstone_keras_best_model.model.keras"
PYTORCH_FILENAME = "ai_capstone_pytorch_state_dict.pth"

CLASS_NAMES = {
    0: "Non-agricultural",
    1: "Agricultural",
}


class ConvNet(nn.Module):
    """Reconstructed PyTorch CNN architecture for the saved state dictionary."""

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1024, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


@lru_cache(maxsize=1)
def load_keras_model():
    """Download and cache the pretrained Keras model."""
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=KERAS_FILENAME,
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )


@lru_cache(maxsize=1)
def load_pytorch_model():
    """Download and cache the pretrained PyTorch model."""
    device = torch.device("cpu")

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=PYTORCH_FILENAME,
    )

    model = ConvNet(num_classes=2)

    # Load the weights on CPU for compatibility with hosted deployments.
    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        state_dict = torch.load(
            model_path,
            map_location=device,
        )

    # Support either a raw state dictionary or a wrapped checkpoint.
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Do not silently ignore architecture/weight mismatches.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def predict_keras(image_source):
    """Return the Keras model's class and probabilities."""
    model = load_keras_model()
    image_array = preprocess_for_keras(image_source)

    # The model returns one sigmoid probability for the agricultural class.
    agricultural_probability = float(
        np.asarray(model.predict(image_array, verbose=0)).reshape(-1)[0]
    )

    probabilities = {
        "Non-agricultural": 1.0 - agricultural_probability,
        "Agricultural": agricultural_probability,
    }

    predicted_class = int(agricultural_probability >= 0.5)

    return {
        "model": "TensorFlow / Keras",
        "class_id": predicted_class,
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": probabilities[CLASS_NAMES[predicted_class]],
        "probabilities": probabilities,
    }


def predict_pytorch(image_source):
    """Return the PyTorch model's class and probabilities."""
    model = load_pytorch_model()
    image_array = preprocess_for_pytorch(image_source)

    image_tensor = torch.from_numpy(image_array).float()

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities_tensor = F.softmax(logits, dim=1)[0]

    probabilities_array = probabilities_tensor.cpu().numpy()

    predicted_class = int(np.argmax(probabilities_array))

    probabilities = {
        "Non-agricultural": float(probabilities_array[0]),
        "Agricultural": float(probabilities_array[1]),
    }

    return {
        "model": "PyTorch",
        "class_id": predicted_class,
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": probabilities[CLASS_NAMES[predicted_class]],
        "probabilities": probabilities,
    }
