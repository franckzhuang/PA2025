import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.title("AI vs Real Landscape - Model Training Dashboard")

st.sidebar.header("Start Training")
model_type = st.sidebar.selectbox(
    "Classification Model",
    ["linear", "svm", "mlp", "kmeans"]
)

params = {}

params["max_images_per_class"] = st.sidebar.number_input(
    "Number of samples per class", min_value=2, max_value=10000, value=90, step=1
)

if model_type == "linear":
    params["learning_rate"] = st.sidebar.number_input(
        "Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f"
    )
    params["max_iterations"] = st.sidebar.number_input(
        "Max Iterations", min_value=1, max_value=10000, value=1000, step=1
    )
elif model_type == "svm":
    params["C"] = st.sidebar.number_input(
        "Regularization parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, format="%.2f"
    )
    params["kernel"] = st.sidebar.selectbox(
        "Kernel", ["linear", "rbf"]
    )

    if params["kernel"] == "rbf":
        params["gamma"] = st.sidebar.number_input(
            "Gamma (for RBF kernel)", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f"
        )

elif model_type == "mlp":
    params["learning_rate"] = st.sidebar.number_input(
        "Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f"
    )

    params["epoch"] = st.sidebar.number_input(
        "Epochs", min_value=1, value=1, step=1,
    )

    params["hidden_layer_sizes"] = st.sidebar.text_input(
        "Hidden Layer Sizes (comma-separated)", value="64,32"
    )

elif model_type == "kmeans":
    params["n_clusters"] = st.sidebar.number_input(
        "Number of clusters", min_value=2, max_value=20, value=2, step=1
    )
    params["max_iterations"] = st.sidebar.number_input(
        "Max Iterations", min_value=1, max_value=10000, value=300, step=1
    )

if st.sidebar.button("Start Training"):
    payload = {
        "model_type": model_type,
    }
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = [
            int(x) for x in params["hidden_layer_sizes"].split(",") if x.strip().isdigit()
        ]
    payload.update(params)

    with st.spinner("Submitting training job..."):
        try:
            r = requests.post(f"{API_URL}/train", json=payload)
            r.raise_for_status()
            job_id = r.json()["job_id"]
            st.session_state["job_id"] = job_id
            st.success(f"Training started! Job ID: {job_id}")
        except Exception as e:
            st.error(f"Failed to start training job. {e}")

job_id = st.session_state.get("job_id", None)
if job_id:
    st.info(f"Polling job status for ID: {job_id}")
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.progress(0)

    for progress in range(100):
        try:
            r = requests.get(f"{API_URL}/train/status/{job_id}")
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            metrics = data.get("metrics")
            error = data.get("error")
            status_placeholder.write(f"Status: **{status}**")
            if status == "finished":
                progress_bar.progress(100)
                if metrics:
                    st.success("Training completed!")
                    metrics_placeholder.json(metrics)
                break
            elif status == "failed":
                st.error(f"Training failed: {error}")
                break
            else:
                progress_bar.progress(progress + 1)
        except Exception as e:
            st.error(f"Error polling job status: {e}")
            break
        time.sleep(1)
    else:
        st.warning("Still waiting... You can refresh or check status again.")

    if st.button("Check Status Again"):
        st.experimental_rerun()
else:
    st.info("Start a training job from the sidebar.")

