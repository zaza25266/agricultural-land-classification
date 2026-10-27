
"""Image preprocessing for the Keras and PyTorch classifiers."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = (64, 64)

# ImageNet normalization used by the PyTorch model.
PYTORCH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PYTORCH_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



def load_rgb_image(image_source):
    """Load an image from a path, file, or existing PIL Image."""

    if isinstance(image_source, Image.Image):
        # It is already a PIL image.
        image = image_source.copy()

    elif isinstance(image_source, (str, Path)):
        # Load from a file path.
        with Image.open(image_source) as opened_image:
            image = opened_image.copy()

    else:
        # Load from an uploaded file or file-like object.
        if hasattr(image_source, "seek"):
            image_source.seek(0)

        with Image.open(image_source) as opened_image:
            image = opened_image.copy()

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def preprocess_for_keras(image_source):
    """
    Prepare an image for the Keras model.

    Returns:
        NumPy array with shape (1, 64, 64, 3), values scaled to [0, 1].

    Note:
        The training documentation specifies nearest-neighbor resizing.
    """
    image = load_rgb_image(image_source)
    image = image.resize(IMAGE_SIZE, Image.Resampling.NEAREST)

    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def preprocess_for_pytorch(image_source):
    """
    Prepare an image for the PyTorch model.

    Returns:
        NumPy array with shape (1, 3, 64, 64), ImageNet-normalized.
    """
    image = load_rgb_image(image_source)
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0

    # Normalize each RGB channel using ImageNet statistics.
    array = (array - PYTORCH_MEAN) / PYTORCH_STD

    # Convert HWC (height, width, channels) to CHW.
    array = np.transpose(array, (2, 0, 1))

    # Add batch dimension: CHW -> NCHW.
    return np.expand_dims(array, axis=0)
