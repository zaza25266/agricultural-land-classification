
"""Streamlit application for agricultural land classification."""

import streamlit as st

from src.inference import (
    load_keras_model,
    load_pytorch_model,
    predict_keras,
    predict_pytorch,
)
from src.preprocessing import load_rgb_image


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Agricultural Land Classification",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Agricultural Land Classification")

st.write(
    "Classify satellite imagery as agricultural or non-agricultural "
    "using trained CNN models built with TensorFlow/Keras and PyTorch."
)

st.info(
    "This is an image-level classifier. It classifies the uploaded "
    "image tile, not individual pixels or field boundaries."
)


# ------------------------------------------------------------
# Cache model loading
# ------------------------------------------------------------

@st.cache_resource
def get_pytorch_model():
    return load_pytorch_model()


@st.cache_resource
def get_keras_model():
    return load_keras_model()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("Model settings")

    selected_model = st.selectbox(
        "Choose a model",
        [
           
            "TensorFlow / Keras",
            "PyTorch",
            "Compare both models",
        ],
    )

    st.caption(
        "Models are downloaded and loaded when first needed."
    )


# ------------------------------------------------------------
# Image upload
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
    help="The models resize images to 64 × 64 before prediction.",
)


# ------------------------------------------------------------
# Run classification
# ------------------------------------------------------------

if uploaded_file is not None:
    try:
        image = load_rgb_image(uploaded_file)

        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded image")

            st.image(
                image,
                caption=f"Original size: {image.width} × {image.height}",
                use_container_width=True,
            )

        with right:
            st.subheader("Classification")

            if st.button(
                "Classify image",
                type="primary",
                use_container_width=True,
            ):
                results = []

                if selected_model in (
                    "Compare both models",
                    "TensorFlow / Keras",
                ):
                    with st.spinner("Running Keras model..."):
                        keras_model = get_keras_model()

                        result = predict_keras(
                            image,
                            model=keras_model,
                        )

                        results.append(result)

                if selected_model in (
                    "Compare both models",
                    "PyTorch",
                ):
                    with st.spinner("Running PyTorch model..."):
                        pytorch_model = get_pytorch_model()

                        result = predict_pytorch(
                            image,
                            model=pytorch_model,
                        )

                        results.append(result)

                # Display results.
                for result in results:
                    st.markdown(f"### {result['model']}")

                    if result["class_id"] == 1:
                        st.success(
                            f"Prediction: {result['prediction']}"
                        )
                    else:
                        st.warning(
                            f"Prediction: {result['prediction']}"
                        )

                    st.metric(
                        "Predicted-class confidence",
                        f"{result['confidence']:.2%}",
                    )

                    st.write("Class probabilities")

                    st.progress(
                        max(
                            0.0,
                            min(
                                1.0,
                                result["probabilities"]["Agricultural"],
                            ),
                        ),
                        text=(
                            "Agricultural: "
                            f"{result['probabilities']['Agricultural']:.2%}"
                        ),
                    )

                    st.caption(
                        "Non-agricultural probability: "
                        f"{result['probabilities']['Non-agricultural']:.2%}"
                    )

                # Compare model predictions.
                if len(results) == 2:
                    st.subheader("Model agreement")

                    if results[0]["class_id"] == results[1]["class_id"]:
                        st.success(
                            f"Both models predict "
                            f"**{results[0]['prediction']}**."
                        )
                    else:
                        st.error(
                            "The models disagree. Treat the prediction "
                            "as uncertain."
                        )

                st.caption(
                    "Confidence is the model's output probability, "
                    "not a guarantee of real-world correctness."
                )

    except Exception as exc:
        st.error("Prediction failed.")

        with st.expander("Technical details"):
            st.exception(exc)

else:
    st.write("Upload a satellite image to get started.")


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Educational AI project · PyTorch · TensorFlow/Keras · "
    "Hugging Face · Streamlit"
)
