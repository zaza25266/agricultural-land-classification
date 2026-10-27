
# src/inference.py
# Loads trained PyTorch and Keras models and predicts land type

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download

from src.preprocessing import (
    load_rgb_image,
    preprocess_for_pytorch,
    preprocess_for_keras,
)


# 1. Configuration


HF_REPO_ID = "ZubairAli25266/agri-land-classification-models"

PYTORCH_FILENAME = "ai_capstone_pytorch_state_dict.pth"
KERAS_FILENAME = "ai_capstone_keras_best_model.model.keras"

CLASS_NAMES = {
    0: "Non-agricultural",
    1: "Agricultural",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. PyTorch model architecture
# Layer indices match the saved checkpoint

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = nn.Sequential(
            # Convolution block 1
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # 0
            nn.ReLU(),                                  # 1
            nn.MaxPool2d(2),                            # 2
            nn.BatchNorm2d(32),                         # 3

            # Convolution block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2), # 4
            nn.ReLU(),                                  # 5
            nn.MaxPool2d(2),                            # 6
            nn.BatchNorm2d(64),                         # 7

            # Convolution block 3
            nn.Conv2d(64, 128, kernel_size=5, padding=2), # 8
            nn.ReLU(),                                   # 9
            nn.MaxPool2d(2),                             # 10
            nn.BatchNorm2d(128),                          # 11

            # Convolution block 4
            nn.Conv2d(128, 256, kernel_size=5, padding=2), # 12
            nn.ReLU(),                                     # 13
            nn.MaxPool2d(2),                               # 14
            nn.BatchNorm2d(256),                           # 15

            # Convolution block 5
            nn.Conv2d(256, 512, kernel_size=5, padding=2), # 16
            nn.ReLU(),                                     # 17
            nn.MaxPool2d(2),                               # 18
            nn.BatchNorm2d(512),                           # 19

            # Convolution block 6
            nn.Conv2d(512, 1024, kernel_size=5, padding=2), # 20
            nn.ReLU(),                                      # 21
            nn.MaxPool2d(2),                                # 22
            nn.BatchNorm2d(1024),                            # 23

            # Classification head
            nn.AdaptiveAvgPool2d((1, 1)), # 24
            nn.Flatten(),                 # 25
            nn.Linear(1024, 2048),         # 26
            nn.ReLU(),                    # 27
            nn.BatchNorm1d(2048),          # 28
            nn.Dropout(0.4),               # 29
            nn.Linear(2048, num_classes),  # 30
        )

    def forward(self, x):
        return self.model(x)

# 3. Load the PyTorch model

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

    # Fail explicitly if the architecture does not match.
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


# 4. Load the Keras model

def load_keras_model():
    """Download and load the trained Keras model."""

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=KERAS_FILENAME,
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    return model



# 5. Prepare the input image


def prepare_image(image):
    """
    Accept a PIL image or image path and return an RGB PIL image.
    """

    if isinstance(image, (str, bytes)):
        image = Image.open(image)

    elif not isinstance(image, Image.Image):
        raise TypeError(
            "Input must be a PIL image or an image file path."
        )

    return image.convert("RGB")


# 6. PyTorch prediction


def predict_pytorch(model, image):
    """
    Predict whether an image contains agricultural land.

    Returns:
        dict with predicted class, confidence, and probabilities.
    """

    image = prepare_image(image)

    # Use the project's existing preprocessing function.
    processed = preprocess_for_pytorch(image)

    # Support either a NumPy array or a torch tensor.
    if isinstance(processed, torch.Tensor):
        tensor = processed
    else:
        tensor = torch.as_tensor(
            np.asarray(processed),
            dtype=torch.float32,
        )

    # Convert HWC image layout to CHW if necessary.
    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)

    # Add batch dimension: CHW -> NCHW.
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device=DEVICE, dtype=torch.float32)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_class = int(torch.argmax(probabilities).item())

    probability_values = probabilities.cpu().tolist()

    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": float(probability_values[predicted_class]),
        "probabilities": {
            CLASS_NAMES[i]: float(probability_values[i])
            for i in range(len(probability_values))
        },
    }


# 7. Keras prediction

def predict_keras(model, image):
    """
    Predict whether an image contains agricultural land.

    Assumes the Keras model returns one sigmoid probability
    for the Agricultural class.
    """

    image = prepare_image(image)

    processed = preprocess_for_keras(image)
    array = np.asarray(processed, dtype=np.float32)

    # Add batch dimension if preprocessing returned one image.
    if array.ndim == 3:
        array = np.expand_dims(array, axis=0)

    prediction = np.asarray(
        model.predict(array, verbose=0)
    )

    # Handle a single sigmoid output.
    if prediction.size == 1:
        agricultural_probability = float(prediction.reshape(-1)[0])

        probabilities = [
            1.0 - agricultural_probability,
            agricultural_probability,
        ]

    # Handle a two-class softmax output.
    elif prediction.shape[-1] == 2:
        probabilities = prediction.reshape(-1, 2)[0].tolist()

    else:
        raise ValueError(
            f"Unexpected Keras output shape: {prediction.shape}. "
            "Expected one sigmoid output or two class probabilities."
        )

    predicted_class = int(np.argmax(probabilities))

    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": float(probabilities[predicted_class]),
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(2)
        },
    }
