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
if "chart_data" not in st.session_state:
    st.session_state.chart_data = []
if "blind_progress" not in st.session_state:
    st.session_state.blind_progress = 0

st.title("🏞️ AI vs Real Landscape - Training Dashboard")

with st.sidebar:
    st.header("🛠️ Configuration")

    with st.expander("1. Data Settings", expanded=True):
        max_samples = st.number_input(
            "Max samples per class", min_value=2, max_value=10000, value=90, step=1
        )
        img_w = st.number_input(
            "Image width", min_value=16, max_value=1024, value=32, step=1
        )
        img_h = st.number_input(
            "Image height", min_value=16, max_value=1024, value=32, step=1
        )

    with st.expander("2. Model Hyperparameters", expanded=True):
        model_type = st.selectbox(
            "Model Type", ["linear_classification", "svm", "mlp", "kmeans"]
        )

        params = {
            "max_images_per_class": max_samples,
            "image_width": img_w,
            "image_height": img_h,
        }
        if model_type == "linear_classification":
            params.update({
                "learning_rate": st.number_input(
                    "Learning Rate", min_value=1e-4, max_value=1.0, value=0.01,
                ),
                "max_iterations": st.number_input(
                    "Max Iterations", min_value=1, max_value=100000, value=1000, step=1
                ),
            })
        elif model_type == "svm":
            params.update({
                "C": st.number_input(
                    "Regularization (C)", min_value=0.01, max_value=100.0, value=1.0, format="%.2f"
                ),
                "kernel": st.selectbox("Kernel", ["linear", "rbf"]),
            })
            if params["kernel"] == "rbf":
                params["gamma"] = st.number_input(
                    "Gamma (RBF)", min_value=1e-4, max_value=1.0, value=0.01, format="%.4f"
                )
        elif model_type == "mlp":
            params.update({
                "learning_rate": st.number_input(
                    "Learning Rate", min_value=1e-4, max_value=1.0, value=0.01, format="%.4f"
                ),
                "epochs": st.number_input(
                    "Epochs", min_value=1, max_value=10000, value=10, step=1
                ),
                "hidden_layer_sizes": st.text_input(
                    "Hidden layers (comma-separated)", value="64,32"
                ),
            })
        elif model_type == "kmeans":
            params.update({
                "n_clusters": st.number_input(
                    "Clusters", min_value=2, max_value=20, value=2, step=1
                ),
                "max_iterations": st.number_input(
                    "Max Iterations", min_value=1, max_value=10000, value=300, step=1
                ),
            })

    start_btn = st.button("🚀 Start Training")

if start_btn:
    payload = params.copy()
    if model_type == "mlp":
        payload["hidden_layer_sizes"] = [int(x) for x in params["hidden_layer_sizes"].split(",") if x.strip().isdigit()]

    try:
        st.session_state.job_id = client.start_training(model_type, payload)
        st.session_state.active_polling = True
        st.session_state.chart_data = []
        st.session_state.blind_progress = 0
        st.success(f"Training started! Job ID: {st.session_state.job_id}")
    except Exception as e:
        st.error(f"Failed to start training: {e}")

if st.session_state.active_polling and st.session_state.job_id:
    spinner = st.spinner("🔄 Training in progress…")
    status_text = st.empty()
    metrics_chart = st.empty()
    progress_bar = st.progress(0)

    with spinner:
        while st.session_state.active_polling:
            try:
                resp = client.get_status(st.session_state.job_id)
                status = resp.get("status")
                metrics = resp.get("metrics", {})
                pct = resp.get("progress", None)

                status_text.markdown(f"**Status:** {status}")
                if pct is not None:
                    st.session_state.blind_progress = pct
                else:
                    st.session_state.blind_progress = min(st.session_state.blind_progress + 5, 100)
                progress_bar.progress(st.session_state.blind_progress)

                if "epoch" in metrics:
                    st.session_state.chart_data.append(metrics)
                    df = pd.DataFrame(st.session_state.chart_data)
                    metrics_chart.line_chart(df.set_index("epoch")[ [k for k in df.columns if k != "epoch"] ])

                if status == "SUCCESS":
                    st.session_state.active_polling = False
                    st.balloons()
                    status_text.success("✅ Training completed successfully!")

                    if metrics:
                        rename_map = {
                            'len_real_images': 'Real Images',
                            'len_ai_images': 'AI Images',
                            'train_samples': 'Train Samples',
                            'test_samples': 'Test Samples',
                            'train_accuracy': 'Train Accuracy',
                            'test_accuracy': 'Test Accuracy'
                        }

                        col1, col2 = st.columns(2)

                        with col1:
                            counts = {
                                rename_map[k]: metrics[k]
                                for k in ['len_real_images', 'len_ai_images']
                                if k in metrics
                            }
                            if counts:
                                df_counts = pd.DataFrame({'count': list(counts.values())}, index=list(counts.keys()))
                                fig1 = px.pie(
                                    df_counts, names=df_counts.index, values='count',
                                    title='📊 Image Source Distribution'
                                )
                                fig1.update_traces(textinfo='value+label')
                                st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            samples = {
                                rename_map[k]: metrics[k]
                                for k in ['train_samples', 'test_samples']
                                if k in metrics
                            }
                            if samples:
                                df_samples = pd.DataFrame({'count': list(samples.values())}, index=list(samples.keys()))
                                fig2 = px.pie(
                                    df_samples, names=df_samples.index, values='count',
                                    title='📚 Data Split Distribution'
                                )
                                fig2.update_traces(textinfo='value+label')
                                st.plotly_chart(fig2, use_container_width=True)

                        st.divider()

                        col3, col4, col5 = st.columns(3)

                        with col3:
                            train_acc = metrics.get('train_accuracy', 0)
                            st.metric(
                                label="🎯 Train Accuracy",
                                value=f"{train_acc:.2f}%"
                            )

                        with col4:
                            test_acc = metrics.get('test_accuracy', 0)
                            st.metric(
                                label="🧪 Test Accuracy",
                                value=f"{test_acc:.2f}%",
                                help="Accuracy on unseen data. This is the most important metric."
                            )

                        with col5:
                            duration = metrics.get('training_duration', 0)
                            st.metric(
                                label="⏱️ Training Duration",
                                value=f"{duration:.2f} s",
                                help="Total time taken for the model training."
                            )

                    else:
                        st.write("No metrics to display.")


                    params_export = json.loads(resp.get("params", "{}"))
                    if params_export:
                        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = f"{dt}_{model_type}_params.json"
                        st.download_button(
                            "📥 Download Model Params",
                            data=json.dumps(params_export, indent=2).encode(),
                            file_name=fname,
                            mime="application/json"
                        )
                        st.info("💡 If you want to save this model to the database, please go to the **History** page.")
                    break
                if status == "FAILURE":
                    st.session_state.active_polling = False
                    err = resp.get("error", "Unknown error")
                    status_text.error(f"Training failed: {err}")
                    break

                time.sleep(1)
            except Exception as e:
                st.session_state.active_polling = False
                status_text.error(f"Error polling status: {e}")
                break

