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

with st.sidebar:
    st.header("Settings")

    models = client.get_models()
    if not models:
        st.error("No models available.")
        st.stop()

    model_type = st.selectbox("Model Type", list(models.keys()))
    model_name = st.selectbox("Model Name", models[model_type])

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    detect_btn = st.button("Run Detection")

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
                    # st.write("🔍 Payload :", {
                    #     "model_type": model_type,
                    #     "model_name": model_name,
                    #     "input_data": (np.array(img.resize((32, 32))).astype(np.float32) / 255.0).flatten().tolist()[:10],  # Affiche juste les 10 premiers éléments
                    #     "total_length": len((np.array(img.resize((32, 32))).astype(np.float32) / 255.0).flatten().tolist())
                    # })
                    
                    result = client.evaluate_model(
                        model_type=model_type,
                        model_name=model_name,
                        input_data = (np.array(img.resize((32, 32))).astype(np.float32) / 255.0).flatten().tolist()
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
