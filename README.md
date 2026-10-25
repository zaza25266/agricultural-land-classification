# Agricultural Land Classification from Satellite Imagery

A deep learning project that classifies satellite image tiles into **agricultural** and **non-agricultural** categories using convolutional neural networks (CNNs) implemented in both **PyTorch** and **TensorFlow/Keras**.

The project includes model training notebooks, a comparative analysis, pretrained model checkpoints hosted on Hugging Face, and a Streamlit application for image-level inference.

## Project Overview

Agricultural land classification from satellite imagery can support land-use analysis, agricultural monitoring, and geospatial research. This project explores a binary image-classification approach using RGB satellite image tiles.

**Classification labels**

- `0` — Non-agricultural
- `1` — Agricultural

**Important:** This application classifies an entire image tile. It does not perform pixel-level segmentation, identify field boundaries, or independently verify land use.

## Project Structure

```text
agricultural-land-classification/
├── notebooks/
│   ├── Train_and_Evaluate_a_Keras-Based_Classifier.ipynb
│   ├── train_and_evalute_a_PyTorch-Based_Classifier.ipynb
│   └── Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── inference.py
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Dataset

The project dataset contains **6,000 RGB satellite image tiles**, organized into two balanced classes.

| Property | Description |
|---|---|
| Total images | 6,000 |
| Agricultural | 3,000 |
| Non-agricultural | 3,000 |
| Image dimensions | 64 × 64 pixels |
| Channels | 3 — RGB |
| Training split | 4,800 images (80%) |
| Validation split | 1,200 images (20%) |

The project documentation describes the imagery as Sentinel-2 Level-2A tiles.

The dataset is organized into class-specific directories for image classification. Dataset access and licensing should be checked before redistribution.

## Models

### TensorFlow / Keras

The Keras classifier uses four convolutional blocks followed by global average pooling and a dense classifier.

- Convolution filters: 32, 64, 128, and 256
- Convolution kernel: 5 × 5
- Activation: ReLU
- Downsampling: Max pooling
- Regularization: Batch normalization and dropout
- Output: One sigmoid probability for the agricultural class
- Loss: Binary cross-entropy

**Documented parameter count:** 1,273,729.

### PyTorch

The PyTorch classifier uses six convolutional blocks followed by adaptive average pooling and a dense classifier.

- Convolution filters: 32, 64, 128, 256, 512, and 1024
- Convolution kernel: 5 × 5
- Activation: ReLU
- Downsampling: Max pooling
- Regularization: Batch normalization and dropout
- Output: Two logits, converted to class probabilities using softmax
- Loss: Cross-entropy

**Documented parameter count:** approximately 19.6 million.

### Training Configuration

| Setting | Value |
|---|---|
| Epochs | 10 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| PyTorch random seed | 42 |
| Keras random seed | 7331 |

The training notebooks contain the model implementations and evaluation workflow. Refer to them for the exact training and validation procedures.

## Model Weights

Pretrained models are hosted separately to keep the GitHub repository lightweight.

**Hugging Face model repository:**  
https://huggingface.co/ZubairAli25266/agri-land-classification-models

Available checkpoints:

- `ai_capstone_keras_best_model.model.keras`
- `ai_capstone_pytorch_state_dict.pth`

The Keras checkpoint contains a saved Keras model. The PyTorch checkpoint contains a state dictionary and requires the corresponding model architecture to load its weights.

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/zaza25266/agricultural-land-classification.git
cd agricultural-land-classification
```

### 2. Install dependencies

Use a supported Python environment, then run:

```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit

```bash
streamlit run app.py
```

The first inference request downloads the selected pretrained model from Hugging Face. An internet connection is required.

## Using the Web Application

1. Open the deployed Streamlit application.
2. Upload an RGB satellite image.
3. Select TensorFlow/Keras, PyTorch, or both models.
4. Run classification.
5. Review the predicted class, output probability, and model agreement.

For the closest match to the documented training data, use compatible 64 × 64 RGB image tiles.

## Evaluation and Results

The project notes report the following comparison across **all 6,000 dataset images**:

| Metric | TensorFlow / Keras | PyTorch |
|---|---:|---:|
| Accuracy | 99.25% | 99.88% |
| Agricultural precision | 100.00% | 99.83% |
| Agricultural recall | 98.50% | 99.93% |
| Agricultural F1-score | 99.24% | 99.88% |
| ROC-AUC | 1.0000 | 1.0000 |
| Log loss | 0.0247 | 0.0024 |
| Misclassified images | 45 | 7 |

**Evaluation limitation:** These figures were calculated over the full dataset, including training images. They are therefore not independent held-out test results and may overstate generalization performance. They should be interpreted as a full-dataset comparison, not as proof of real-world accuracy.

A stronger evaluation should use a separate test set that was never used for training, checkpoint selection, or hyperparameter tuning. If images from nearby geographic areas are correlated, a geographically separated test set would provide a more meaningful assessment.

The training documentation also reports a best PyTorch validation loss of 0.0042 and validation accuracy of 99.92% at epoch 5, and a best Keras validation accuracy of 98.92% at epoch 7. The notebooks should be consulted for the precise evaluation procedures and augmentation settings.

## Preprocessing

The inference pipeline uses separate preprocessing functions for the two models.

**Keras**
- Convert to RGB.
- Resize to 64 × 64 using nearest-neighbor interpolation.
- Scale pixel values to the range [0, 1].
- Add a batch dimension in channels-last format.

**PyTorch**
- Convert to RGB.
- Resize to 64 × 64 using bilinear interpolation.
- Scale pixel values to [0, 1].
- Normalize RGB channels using ImageNet mean and standard deviation.
- Convert to channels-first format and add a batch dimension.

Preprocessing must match the actual training pipeline. Differences between training and inference preprocessing can affect predictions.

## Limitations

- Binary classification only; no pixel-level segmentation.
- Predictions depend on image quality, geographic context, acquisition conditions, and similarity to the training data.
- Model confidence is not a guarantee of correctness.
- Full-dataset comparison metrics include training samples and do not establish independent test performance.
- The model architecture and preprocessing implementation must match the saved checkpoints.
- Hosting both TensorFlow and PyTorch can increase deployment startup time and memory usage.

## Future Improvements

- Establish a leakage-resistant, independent test set.
- Evaluate generalization across geographic regions and seasons.
- Add confusion matrices, precision-recall curves, and calibration analysis.
- Explore multispectral inputs rather than RGB alone.
- Investigate segmentation for pixel-level agricultural mapping.
- Improve model loading, caching, and deployment monitoring.

## Technologies

- Python
- PyTorch
- TensorFlow / Keras
- NumPy
- Pillow
- Hugging Face Hub
- Streamlit
- Jupyter Notebook

## Author

**Zubair Ali**

- GitHub: https://github.com/zaza25266
- Project repository: https://github.com/zaza25266/agricultural-land-classification
- Model repository: https://huggingface.co/ZubairAli25266/agri-land-classification-models

