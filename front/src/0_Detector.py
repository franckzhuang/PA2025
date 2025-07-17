import streamlit as st
import numpy as np
from PIL import Image
from utils import ApiClient

client = ApiClient()

st.set_page_config(
    page_title="AGID - Anti Generated Image Detection",
    layout="wide",
    page_icon="🔎"
)

st.title("🔎 AGID")
st.subheader("Anti Generated Image Detection")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Settings")

    models = client.get_models()
    if not models:
        st.error("No models available.")
        st.stop()

    model_type = st.selectbox("Model Type", list(models.keys()))
    model_name = st.selectbox("Model Name", models[model_type])

    show_details_btn = st.button("Show Model Details")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    detect_btn = st.button("🚀 Run Detection")

@st.dialog("Model Details", width="large")
def show_model_details():
    with st.spinner("Fetching model details..."):
        try:
            details, params = client.get_model_details(model_name)

            st.write("## General Information")
            st.write(f"**Model Name:** {details.get('model_name')}")
            st.write(f"**Model Type:** {details.get('model_type')}")
            st.write(f"**Created At:** {details.get('created_at')}")

            st.divider()

            st.write("## Hyperparameters")
            st.json(details.get("job", {}).get("hyperparameters"))

            st.write("## Image Configuration")
            st.json(details.get("job", {}).get("image_config"))

            st.write("## Metrics")
            st.json(details.get("job", {}).get("metrics"))

            st.write("## Model Parameters")
            st.json(params)

        except Exception as e:
            st.error(f"Error fetching model details: {e}")


if show_details_btn:
    show_model_details()

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    max_height = 600
    if img.height > max_height:
        ratio = max_height / img.height
        new_size = (int(img.width * ratio), max_height)
        img_display = img.resize(new_size)
    else:
        img_display = img

    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        result_placeholder = st.empty()

        st.image(
            img_display,
            caption="Uploaded Image Preview",
            output_format="PNG"
        )

        if detect_btn:
            with st.spinner("Evaluating..."):
                try:
                    img_array = np.array(img.convert("RGB"), dtype=np.float32)
                    result = client.evaluate_model(
                        model_type=model_type,
                        model_name=model_name,
                        input_data=img_array.tolist()
                    )

                    pred = result.get("prediction")
                    score = float(pred[0]) if isinstance(pred, list) else float(pred)

                    if score < 0:
                        result_placeholder.error("🤖 This image is likely AI-generated.")
                        st.toast("Detected as AI-generated", icon="🤖")
                    else:
                        result_placeholder.success("✅ This looks like a real photo.")
                        st.toast("Detected as real photo", icon="✅")
                        st.balloons()
                except Exception as e:
                    result_placeholder.error(f"Error during evaluation: {e}")

else:
    if detect_btn:
        st.warning("Please upload an image before running detection.")
