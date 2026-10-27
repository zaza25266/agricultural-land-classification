
"""Image preprocessing for Keras and PyTorch classifiers."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = (64, 64)

PYTORCH_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)

PYTORCH_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)


def load_rgb_image(image_source):
    """Load an image from a path, file object, or PIL image."""

    if isinstance(image_source, Image.Image):
        image = image_source.copy()

    elif isinstance(image_source, (str, Path)):
        with Image.open(image_source) as opened_image:
            image = opened_image.copy()

    else:
        if hasattr(image_source, "seek"):
            image_source.seek(0)

        with Image.open(image_source) as opened_image:
            image = opened_image.copy()

    image = ImageOps.exif_transpose(image)

    return image.convert("RGB")


def preprocess_for_keras(image_source):
    """
    Prepare an image for Keras.

    Output shape: (1, 64, 64, 3)
    Values: [0, 1]
    """

    image = load_rgb_image(image_source)

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    return np.expand_dims(array, axis=0)


def preprocess_for_pytorch(image_source):
    """
    Prepare an image for PyTorch.

    Output shape: (1, 3, 64, 64)
    Values: ImageNet-normalized
    """

    image = load_rgb_image(image_source)

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    # Normalize the RGB channels.
    array = (array - PYTORCH_MEAN) / PYTORCH_STD

    # HWC -> CHW -> NCHW.
    array = np.transpose(array, (2, 0, 1))

    return np.expand_dims(array, axis=0)
