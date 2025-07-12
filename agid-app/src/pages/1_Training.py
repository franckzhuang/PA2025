import os
import time
import json
from datetime import datetime

import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI vs Real Landscape Trainer", page_icon="🏞️", layout="wide"
)

class ApiClient:
    def __init__(self, base_url, timeout=5, max_retries=3):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = timeout
        adapter = requests.adapters.HTTPAdapter(max_retries=max_retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def start_training(self, model_type, payload):
        url = f"{self.base_url}/train/{model_type}"
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["job_id"]

    def get_status(self, job_id):
        url = f"{self.base_url}/train/status/{job_id}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_history(self):
        url = f"{self.base_url}/train/history"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

client = ApiClient(API_URL)

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
        upload_files = st.file_uploader(
            "Optional: Upload custom dataset (ZIP)", type=["zip"], accept_multiple_files=False
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
                    "Learning Rate", min_value=1e-4, max_value=1.0, value=0.01, format="%.4f"
                ),
                "max_iterations": st.number_input(
                    "Max Iterations", min_value=1, max_value=10000, value=1000, step=1
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
    if upload_files:
        # TODO: To implement
        st.warning("Custom dataset upload is not yet implemented.")

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
                    status_text.success("Training completed successfully!")

                    if metrics:
                        rename_map = {
                            'total_images': 'Total Images',
                            'real_images': 'Real Images',
                            'ai_images': 'AI Images',
                            'train_samples': 'Train Samples',
                            'test_samples': 'Test Samples',
                            'train_accuracy': 'Train Accuracy',
                            'test_accuracy': 'Test Accuracy'
                        }

                        counts = {
                            rename_map[k]: metrics[k]
                            for k in ['real_images', 'ai_images']
                            if k in metrics
                        }
                        samples = {
                            rename_map[k]: metrics[k]
                            for k in ['train_samples', 'test_samples']
                            if k in metrics
                        }
                        accuracies = {
                            rename_map[k]: metrics[k]
                            for k in ['train_accuracy', 'test_accuracy']
                            if k in metrics
                        }

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            df_counts = pd.DataFrame({'count': list(counts.values())}, index=list(counts.keys()))
                            fig1 = px.pie(
                                df_counts, names=df_counts.index, values='count',
                                title='Count Distribution'
                            )
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            df_samples = pd.DataFrame({'count': list(samples.values())}, index=list(samples.keys()))
                            fig2 = px.pie(
                                df_samples, names=df_samples.index, values='count',
                                title='Sample Distribution'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        with col3:
                            df_acc = pd.DataFrame({ 'Accuracy': list(accuracies.values()) }, index=list(accuracies.keys()))
                            st.bar_chart(df_acc)

                    else:
                        st.write("No metrics to display.")

                    st.json(metrics)
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

with st.expander("📜 Training History", expanded=False):
    try:
        history = client.get_history()
        if history:
            df_hist = pd.DataFrame(history)
            st.dataframe(df_hist)
        else:
            st.write("No history available.")
    except Exception:
        st.write("Unable to fetch history.")
