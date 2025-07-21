import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from utils import ApiClient, format_duration
import plotly.express as px

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

            job_details = details.get("job", {})
            hyperparameters = job_details.get("hyperparameters", {})
            image_config = job_details.get("image_config", {})
            metrics = job_details.get("metrics", {})

            st.write(f"### {details.get('model_name')} ({details.get('model_type')})")
            st.caption(f"Job ID: {job_details.get('job_id')}")
            st.divider()

            with st.expander("📈 Metrics", expanded=True):
                if not metrics:
                    st.write("No metrics available for this run.")
                else:
                    # Métriques finales
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0):.2f}%")
                    m_col2.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2f}%")
                    m_col3.metric("Duration", format_duration(metrics.get('training_duration', 0)))

                    st.divider()

                    pie_col1, pie_col2 = st.columns(2)
                    with pie_col1:
                        df_counts = pd.DataFrame({
                            'type': ['Real', 'AI'],
                            'count': [metrics.get('len_real_images', 0), metrics.get('len_ai_images', 0)]
                        })
                        fig1 = px.pie(df_counts, names='type', values='count',
                                      title='📊 Image Source Distribution')
                        fig1.update_traces(textinfo='value+label')
                        st.plotly_chart(fig1, use_container_width=True)

                    with pie_col2:
                        df_samples = pd.DataFrame({
                            'type': ['Train', 'Test'],
                            'count': [metrics.get('train_samples', 0), metrics.get('test_samples', 0)]
                        })
                        fig2 = px.pie(df_samples, names='type', values='count',
                                      title='📚 Data Split Distribution')
                        fig2.update_traces(textinfo='value+label')
                        st.plotly_chart(fig2, use_container_width=True)

                    if 'train_losses' in metrics and isinstance(metrics.get('train_losses'), list):
                        st.divider()
                        loss_col, acc_col = st.columns(2)
                        with loss_col:
                            df_losses = pd.DataFrame(
                                {
                                    "Train Loss": metrics["train_losses"],
                                    "Test Loss": metrics.get("test_losses", []),
                                }
                            )
                            st.line_chart(df_losses)
                            st.caption("Loss Evolution (Train vs Test)")
                        with acc_col:
                            df_accuracies = pd.DataFrame(
                                {
                                    "Train Accuracy": metrics["train_accuracies"],
                                    "Test Accuracy": metrics.get("test_accuracies", []),
                                }
                            )
                            st.line_chart(df_accuracies)
                            st.caption("Accuracy Evolution (Train vs Test)")

            with st.expander("⚙️ Hyperparameters"):
                display_params = {}
                if image_config.get('image_size'):
                    display_params['image_size'] = image_config['image_size']
                if image_config.get('images_per_class'):
                    display_params['images_per_class'] = image_config['images_per_class']

                display_params.update(hyperparameters)

                if not display_params:
                    st.info("No hyperparameters found.")
                else:
                    display_data = {k: str(v) for k, v in display_params.items()}
                    df_params = pd.DataFrame(display_data.items(), columns=["Parameter", "Value"])
                    st.table(df_params)

            with st.expander("🖼️ Images Used"):
                col_train, col_test = st.columns(2)
                training_images = image_config.get('training_images', {})
                test_images = image_config.get('test_images', {})

                with col_train:
                    st.write("**Training Images**")
                    train_real = training_images.get('real')
                    train_ai = training_images.get('ai')
                    if train_real:
                        st.write(f"_Real ({len(train_real)}):_")
                        st.dataframe(train_real, height=150, use_container_width=True)
                    if train_ai:
                        st.write(f"_AI ({len(train_ai)}):_")
                        st.dataframe(train_ai, height=150, use_container_width=True)

                with col_test:
                    st.write("**Test Images**")
                    test_real = test_images.get('real')
                    test_ai = test_images.get('ai')
                    if test_real:
                        st.write(f"_Real ({len(test_real)}):_")
                        st.dataframe(test_real, height=150, use_container_width=True)
                    if test_ai:
                        st.write(f"_AI ({len(test_ai)}):_")
                        st.dataframe(test_ai, height=150, use_container_width=True)

            with st.expander("🧮 Model Parameters (Raw)"):
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
                    _, params = client.get_model_details(model_name)
                    result = client.evaluate_model(
                        model_type=model_type,
                        model_name=model_name,
                        params=params,
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
