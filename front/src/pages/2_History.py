import os
import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
import plotly.express as px

from utils import ApiClient, format_duration


client = ApiClient()

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)
API_URL = os.environ.get("API_URL", "http://localhost:8000")


st.set_page_config(page_title="Training History", page_icon="📜", layout="wide")

st.title("📜 Training History")
st.markdown("Explore, filter and analyze the results of all past training sessions.")

try:
    with st.spinner("🔄 Fetching training history from the API..."):
        history_data = client.get_history()

    if not history_data:
        st.warning("No training history found.")
    else:
        df = pd.json_normalize(history_data, sep=".")

        st.sidebar.header("🔎 Filters")
        model_types = df["model_type"].unique()
        selected_models = st.sidebar.multiselect(
            "Filter by Model Type", options=model_types, default=model_types
        )

        statuses = df["status"].unique()
        selected_statuses = st.sidebar.multiselect(
            "Filter by Status", options=statuses, default=statuses
        )

        filtered_df = df[
            df["model_type"].isin(selected_models)
            & df["status"].isin(selected_statuses)
        ].copy()

        st.header(f"Displaying {len(filtered_df)} of {len(df)} runs")

        display_columns = {
            "created_at": "Date",
            "model_type": "Model",
            "status": "Status",
            "metrics.test_accuracy": "Test Accuracy (%)",
            "model_saved": "Model Saved",
            "job_id": "Job ID",
        }

        cols_to_display = [
            col for col in display_columns.keys() if col in filtered_df.columns
        ]

        if not cols_to_display:
            st.error("The history data is missing key columns to display.")
        else:
            display_df = filtered_df[cols_to_display].rename(columns=display_columns)
            display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if "Test Accuracy (%)" in display_df.columns:
                display_df["Test Accuracy (%)"] = display_df["Test Accuracy (%)"].map(
                    "{:.2f}%".format
                )

            st.dataframe(display_df, use_container_width=True)

            st.divider()
            st.header("🔬 Inspect a Specific Run")

            if filtered_df.empty:
                st.info("No runs to inspect based on current filters.")
            else:
                job_ids = filtered_df["job_id"].tolist()
                selected_job_id = st.selectbox(
                    "Select a Job ID to see full details", options=job_ids
                )

                if selected_job_id:
                    job_details_row = filtered_df[
                        filtered_df["job_id"] == selected_job_id
                    ].iloc[0]

                    st.subheader("Actions")
                    action_cols = st.columns(2)

                    with action_cols[0]:
                        if "params_file" in job_details_row and pd.notna(
                            job_details_row["params_file"]
                        ):
                            try:

                                params_content = client.get_params_for_job(
                                    selected_job_id
                                )
                                job_dict_cleaned = job_details_row.dropna().to_dict()
                                model_type = job_dict_cleaned.get("model_type", "model")
                                export_data = {
                                    "model_type": model_type,
                                    "params": params_content,
                                    "job_details": job_dict_cleaned,
                                }
                                date_str = pd.to_datetime(
                                    job_dict_cleaned.get("created_at")
                                ).strftime("%Y%m%d_%H%M%S")
                                export_filename = st.text_input(
                                    "Enter a name to save the model and job parameters as JSON file.",
                                )
                                if export_filename:
                                    st.download_button(
                                        label="📥 Download Model Export",
                                        data=json.dumps(
                                            export_data, indent=2, default=str
                                        ).encode("utf-8"),
                                        file_name=export_filename + ".json",
                                        mime="application/json",
                                    )
                            except Exception as e:
                                st.error(f"Download failed: {e}")
                        else:
                            st.info("No downloadable parameters.")

                    with action_cols[1]:
                        if "params_file" not in job_details_row or not pd.notna(
                            job_details_row["params_file"]
                        ):
                            st.info("Model cannot be saved.")
                        else:
                            model_name_input = st.text_input(
                                "Enter a unique name to save the model to database",
                            )
                            if st.button("✅ Save Model"):
                                if not model_name_input.strip():
                                    st.error("Please enter a model name before saving.")
                                else:
                                    try:
                                        response = client.save_model(
                                            job_id=selected_job_id, name=model_name_input.strip()
                                        )

                                        if response.get("status") == "created":
                                            st.success("Model saved successfully! 🎉")
                                            st.balloons()
                                        elif response.get("status") == "exists":
                                            st.warning("A model with this name already exists.")
                                        elif response.get("status") == "not_found":
                                            st.error("The associated training job was not found.")
                                        elif response.get("status") == "error":
                                            st.error(
                                                f"Error saving model: {response.get('message')}"
                                            )
                                        else:
                                            st.error(f"Unexpected response: {response}")
                                    except Exception as e:
                                        st.error(f"Failed to save model: {e}")

                    st.divider()

                    with st.expander("📈 Metrics", expanded=True):
                        metrics = {
                            k.replace("metrics.", ""): v
                            for k, v in job_details_row.to_dict().items()
                            if "metrics." in k
                        }
                        if not metrics:
                            st.write("No metrics available for this run.")
                        else:
                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric(
                                "🎯 Train Accuracy",
                                f"{metrics.get('train_accuracy', 0):.2f}%",
                            )
                            m_col2.metric(
                                "🧪 Test Accuracy",
                                f"{metrics.get('test_accuracy', 0):.2f}%",
                            )
                            m_col3.metric(
                                "⏱️ Training duration",
                                format_duration(metrics.get('training_duration', 0)),
                            )
                            st.divider()

                            st.subheader("📂 Data Distribution")
                            col_pie1, col_pie2 = st.columns(2)
                            with col_pie1:
                                df_counts = pd.DataFrame({
                                    'type': ['Real', 'AI'],
                                    'count': [metrics.get('len_real_images', 0), metrics.get('len_ai_images', 0)]
                                })
                                fig1 = px.pie(df_counts, names='type', values='count',
                                              title='📊 Image Source Distribution')
                                fig1.update_traces(textinfo='value+label')
                                st.plotly_chart(fig1, use_container_width=True)

                            with col_pie2:
                                df_samples = pd.DataFrame({
                                    'type': ['Train', 'Test'],
                                    'count': [metrics.get('train_samples', 0), metrics.get('test_samples', 0)]
                                })
                                fig2 = px.pie(df_samples, names='type', values='count',
                                              title='📚 Data Split Distribution')
                                fig2.update_traces(textinfo='value+label')
                                st.plotly_chart(fig2, use_container_width=True)

                            if "train_losses" in metrics and isinstance(
                                metrics.get("train_losses"), list
                            ):
                                st.divider()
                                st.subheader("📉 Training Evolution per Epoch")
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

                        hyperparameters_data = {}
                        if 'image_config.image_size' in job_dict_cleaned:
                            hyperparameters_data['image_size'] = job_dict_cleaned['image_config.image_size']

                        for key, value in job_dict_cleaned.items():
                            if key.startswith('hyperparameters.'):
                                param_name = key.replace('hyperparameters.', '')
                                hyperparameters_data[param_name] = value

                        if not hyperparameters_data:
                            st.info("No hyperparameters found.")
                        else:

                            display_data = {k: str(v) for k, v in hyperparameters_data.items()}
                            df_params = pd.DataFrame(display_data.items(), columns=["Parameter", "Value"])
                            st.table(df_params)

                    with st.expander("🖼️ Images Used"):
                        col_train, col_test = st.columns(2)

                        with col_train:
                            st.write("**Training Images**")
                            train_real = job_dict_cleaned.get('image_config.training_images.real')
                            train_ai = job_dict_cleaned.get('image_config.training_images.ai')

                            if train_real:
                                st.write(f"_Real ({len(train_real)}):_")
                                st.dataframe(train_real, height=150, use_container_width=True)
                            if train_ai:
                                st.write(f"_AI ({len(train_ai)}):_")
                                st.dataframe(train_ai, height=150, use_container_width=True)

                        with col_test:
                            st.write("**Test Images**")
                            test_real = job_dict_cleaned.get('image_config.test_images.real')
                            test_ai = job_dict_cleaned.get('image_config.test_images.ai')

                            if test_real:
                                st.write(f"_Real ({len(test_real)}):_")
                                st.dataframe(test_real, height=150, use_container_width=True)
                            if test_ai:
                                st.write(f"_AI ({len(test_ai)}):_")
                                st.dataframe(test_ai, height=150, use_container_width=True)
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
