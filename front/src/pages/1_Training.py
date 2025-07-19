import os
import time
import json
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px
from utils import ApiClient

client = ApiClient()

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI vs Real Landscape Trainer", page_icon="🏞️", layout="wide"
)

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "active_polling" not in st.session_state:
    st.session_state.active_polling = False
if "last_run_results" not in st.session_state:
    st.session_state.last_run_results = None

st.title("🏞️ AI vs Real Landscape - Training Dashboard")

with st.sidebar:
    st.header("🛠️ Configuration")

    is_training = st.session_state.active_polling


    with st.expander("1. Data Settings", expanded=True):
        max_samples = st.number_input(
            "Samples per class", min_value=2, max_value=9765, value=90, step=1, disabled=is_training
        )
        img_w = st.number_input(
            "Image width", min_value=16, max_value=1024, value=32, step=1, disabled=is_training
        )
        img_h = st.number_input(
            "Image height", min_value=16, max_value=1024, value=32, step=1, disabled=is_training
        )

    with st.expander("2. Model Hyperparameters", expanded=True):
        model_type = st.selectbox(
            "Model Type", ["linear_classification", "svm", "mlp", "rbf"],
            disabled=is_training,
        )

        params = {
            "max_images_per_class": max_samples,
            "image_width": img_w,
            "image_height": img_h,
        }
        if model_type == "linear_classification":
            params.update(
                {
                    "learning_rate": st.number_input(
                        "Learning Rate",
                        min_value=1e-8,
                        max_value=1.0,
                        value=0.01,
                        disabled=is_training,

                    ),
                    "max_iterations": st.number_input(
                        "Max Iterations",
                        min_value=1,
                        max_value=100000,
                        value=1000,
                        step=1,
                        disabled=is_training,
                    ),
                }
            )
        elif model_type == "svm":
            params.update(
                {
                    "C": st.number_input(
                        "Regularization (C)",
                        min_value=0.01,
                        max_value=100.0,
                        value=1.0,
                        disabled=is_training,
                    ),
                    "kernel": st.selectbox("Kernel", ["linear", "rbf"], disabled=is_training),
                }
            )
            if params["kernel"] == "rbf":
                params["gamma"] = st.number_input(
                    "Gamma (RBF)",
                    min_value=1e-4,
                    max_value=1.0,
                    value=0.01,
                    disabled=is_training,
                )
        elif model_type == "mlp":
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-10,
                max_value=1.0,
                value=0.01,
                disabled=is_training,
            )
            epochs = st.number_input(
                "Epochs", min_value=1, max_value=10000, value=10, step=1
            )

            st.divider()

            st.write("**Layer Configuration**")

            num_layers = st.number_input("Number of hidden layers", min_value=1, max_value=10, value=3)

            hidden_layer_sizes = []
            activations = []

            for i in range(num_layers):
                st.write(f"--- Layer {i + 1} ---")
                col1, col2 = st.columns(2)

                with col1:
                    size = col1.number_input(
                        "Neurons",
                        min_value=1,
                        value=32 if i < num_layers - 1 else 1,
                        key=f"layer_size_{i}",
                        disabled=is_training,
                    )
                    hidden_layer_sizes.append(size)

                with col2:
                    activation = col2.selectbox(
                        "Activation",
                        options=["sigmoid", "linear"],
                        index=0,
                        key=f"activation_{i}",
                        disabled = is_training,
                    )
                    activations.append(activation)

            params.update({
                "learning_rate": learning_rate,
                "epochs": epochs,
                "hidden_layer_sizes": hidden_layer_sizes,
                "activations": activations,
            })
        elif model_type == "rbf":
            rbf_type = st.selectbox(
                "RBF Type",
                options=["naive", "kmeans"],
                disabled=is_training
            )

            gamma = st.number_input(
                "Gamma",
                min_value=1e-4,
                max_value=10.0,
                value=0.1,
                disabled=is_training,
            )

            rbf_params = {
                "rbf_type": rbf_type,
                "gamma": gamma
            }

            if rbf_type == "kmeans":
                st.write("*K-Means Specific Parameters*")
                k = st.number_input(
                    "K (Number of Centers)",
                    min_value=2,
                    max_value=100,
                    value=10,
                    step=1,
                    disabled=is_training,
                )
                max_iterations = st.number_input(
                    "Max Iterations",
                    min_value=1,
                    max_value=10000,
                    value=300,
                    step=1,
                    disabled=is_training,
                )
                rbf_params.update({
                    "k": k,
                    "max_iterations": max_iterations
                })

            params.update(rbf_params)

    start_btn = st.button("🚀 Start Training", disabled=is_training)
    st.markdown("---")
    st.markdown("📂 Import Existing Model")

    uploaded_model_file = st.file_uploader(
        "Upload a saved model JSON file",
        type=["json"],
        disabled=is_training,
    )

    import_btn = st.button("📤 Import Model", disabled=is_training)

if import_btn:
    if uploaded_model_file:
        try:
            model_data = json.load(uploaded_model_file)
            res = client.import_model(model_data)

            status = res.get("status")
            if status == "created":
                st.success(
                    f"✅ Model imported successfully!\n\n"
                    f"**Job ID:** `{res.get('job_id')}`\n\n"
                    f"**Model Name:** `{res.get('model_name')}`"
                )
                st.balloons()

            elif status == "exists":
                st.warning(
                    f"⚠️ A training job with this job_id already exists, the model is already implemented.\n\n"
                )
            elif status == "error":
                st.error(f"❌ Error: {res.get('message')}")
            else:
                st.error(f"❌ Unknown response: {res}")
        except Exception as e:
            st.error(f"❌ Exception while importing model:\n\n{e}")
    else:
        st.warning("⚠️ Please upload a JSON file before importing.")


if start_btn:
    payload = params.copy()

    try:
        st.session_state.last_run_results = None

        st.session_state.job_id = client.start_training(model_type, payload)
        st.session_state.active_polling = True
        st.session_state.blind_progress = 0
        st.success(f"Training started! Job ID: {st.session_state.job_id}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to start training: {e}")

if st.session_state.active_polling and st.session_state.job_id:
    with st.spinner("🔄 Training in progress…"):
        while True:
            try:
                resp = client.get_status(st.session_state.job_id)
                status = resp.get("status")
                st.info(f"Status: {status}")

                if status in ["SUCCESS", "FAILURE"]:
                    st.session_state.active_polling = False
                    st.session_state.last_run_results = resp
                    st.rerun()

                time.sleep(2)
            except Exception as e:
                st.session_state.active_polling = False
                st.session_state.last_run_results = {"status": "FAILURE", "error": str(e)}
                st.rerun()

if st.session_state.last_run_results:
    if st.button("🧨 Reset"):
        st.session_state.last_run_results = None
        st.rerun()

    st.divider()

    results = st.session_state.last_run_results
    status = results.get("status")

    if status == "SUCCESS":
        st.success("✅ Training completed successfully!")
        st.balloons()

        metrics = results.get("metrics", {})
        if not metrics:
            st.write("No metrics to display.")
        else:

            st.subheader("📂 Data Distribution")
            col_pie1, col_pie2 = st.columns(2)
            with col_pie1:
                df_counts = pd.DataFrame({
                    'type': ['Real', 'AI'],
                    'count': [metrics.get('len_real_images', 0), metrics.get('len_ai_images', 0)]
                })
                fig1 = px.pie(df_counts, names='type', values='count', title='📊 Image Source Distribution')
                st.plotly_chart(fig1, use_container_width=True)

            with col_pie2:
                df_samples = pd.DataFrame({
                    'type': ['Train', 'Test'],
                    'count': [metrics.get('train_samples', 0), metrics.get('test_samples', 0)]
                })
                fig2 = px.pie(df_samples, names='type', values='count', title='📚 Data Split Distribution')
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()

            st.subheader("📊 Final Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Train Accuracy", f"{metrics.get('train_accuracy', 0):.2f}%")
            col2.metric("🧪 Test Accuracy", f"{metrics.get('test_accuracy', 0):.2f}%")
            col3.metric("⏱️ Training duration", f"{metrics.get('training_duration', 0):.2f}s")

            if 'train_losses' in metrics and 'train_accuracies' in metrics:
                st.divider()
                st.subheader("📈 Training Evolution per Epoch")
                col_loss, col_acc = st.columns(2)

                with col_loss:
                    df_losses = pd.DataFrame({
                        'Train Loss': metrics['train_losses'],
                        'Test Loss': metrics.get('test_losses', [])
                    })
                    st.line_chart(df_losses)
                    st.caption("Loss Evolution (Train vs Test)")

                with col_acc:
                    df_accuracies = pd.DataFrame({
                        'Train Accuracy': metrics['train_accuracies'],
                        'Test Accuracy': metrics.get('test_accuracies', [])
                    })
                    st.line_chart(df_accuracies)
                    st.caption("Accuracy Evolution (Train vs Test)")

            st.divider()

            if results.get("params_file"):
                try:
                    params_content = client.get_params_for_job(st.session_state.job_id)
                    export_data = {
                        "model_type": results.get("model_type"),
                        "params": params_content,
                        "job": results,
                    }
                    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"{dt}_{results.get('model_type', 'model')}_export.json"
                    st.info(
                        "💡 If you want to save this model to the database, please go to the **History** page."
                    )
                    st.download_button(
                        "📥 Download Model Export",
                        data=json.dumps(export_data, indent=2, default=str).encode('utf-8'),
                        file_name=fname,
                        mime="application/json",
                    )
                except Exception as e:
                    st.error(f"Failed to fetch model params for export: {e}")

    elif status == "FAILURE":
        err = results.get("error", "Unknown error")
        st.error(f"Training failed: {err}")
