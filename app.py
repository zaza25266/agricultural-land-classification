
"""Streamlit application for agricultural land classification."""

import streamlit as st
from PIL import Image

from src.inference import predict_keras, predict_pytorch


st.set_page_config(
    page_title="Agricultural Land Classification",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Agricultural Land Classification")
st.write(
    "Classify satellite imagery as agricultural or non-agricultural "
    "using pretrained CNN models built with TensorFlow/Keras and PyTorch."
)

st.info(
    "This is an image-level classifier. Predictions apply to the uploaded "
    "image tile, not individual pixels or specific field boundaries."
)

with st.sidebar:
    st.header("Model settings")
    selected_model = st.selectbox(
        "Choose a model",
        [
            "Compare both models",
            "TensorFlow / Keras",
            "PyTorch",
        ],
    )

    st.caption("Models are downloaded from Hugging Face when first needed.")

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
    help="For the closest match to training data, use a 64 × 64 RGB satellite tile.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Uploaded image")
            st.image(image, caption=f"Image dimensions: {image.width} × {image.height}",
                     use_container_width=True)

        with right:
            st.subheader("Classification")

            if st.button("Classify image", type="primary", use_container_width=True):
                results = []

                if selected_model in (
                    "Compare both models",
                    "TensorFlow / Keras",
                ):
                    with st.spinner("Running TensorFlow / Keras model..."):
                        results.append(predict_keras(image.copy()))

                if selected_model in (
                    "Compare both models",
                    "PyTorch",
                ):
                    with st.spinner("Running PyTorch model..."):
                        results.append(predict_pytorch(image.copy()))

                for result in results:
                    st.markdown(f"### {result['model']}")

                    if result["class_id"] == 1:
                        st.success(f"Prediction: {result['prediction']}")
                    else:
                        st.warning(f"Prediction: {result['prediction']}")

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

                if len(results) == 2:
                    st.subheader("Model agreement")

                    if results[0]["class_id"] == results[1]["class_id"]:
                        st.success(
                            f"Both models predict **{results[0]['prediction']}**."
                        )
                    else:
                        st.error(
                            "The models disagree. Review the image and "
                            "treat this prediction as uncertain."
                        )

                st.caption(
                    "Confidence is the model's output probability, not a "
                    "guarantee of real-world correctness."
                )

    except Exception as exc:
        st.error(
            "The image could not be processed or a model failed to load. "
            "Check the deployment logs for the underlying error."
        )
        with st.expander("Technical details"):
            st.code(str(exc))

else:
    st.write(
        "Upload a satellite image to get started. The application will "
        "load only the model or models you select."
    )

st.divider()
st.caption(
    "Educational AI project · PyTorch · TensorFlow/Keras · "
    "Hugging Face · Streamlit"
)
