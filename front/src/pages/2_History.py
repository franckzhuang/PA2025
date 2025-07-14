import os
import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv

from utils import ApiClient
client = ApiClient()

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)
API_URL = os.environ.get("API_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Training History",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Training History")
st.markdown("Explore, filter and analyze the results of all past training sessions.")

try:
    with st.spinner("🔄 Fetching training history from the API..."):
        history_data = client.get_history()

    if not history_data:
        st.warning("No training history found.")
    else:
        df = pd.json_normalize(history_data, sep='.')

        st.sidebar.header("🔎 Filters")
        model_types = df['model_type'].unique()
        selected_models = st.sidebar.multiselect(
            "Filter by Model Type", options=model_types, default=model_types
        )

        statuses = df['status'].unique()
        selected_statuses = st.sidebar.multiselect(
            "Filter by Status", options=statuses, default=statuses
        )

        filtered_df = df[
            df['model_type'].isin(selected_models) &
            df['status'].isin(selected_statuses)
            ].copy()

        st.header(f"Displaying {len(filtered_df)} of {len(df)} runs")

        display_columns = {
            'created_at': 'Date',
            'model_type': 'Model',
            'status': 'Status',
            'metrics.test_accuracy': 'Test Accuracy (%)',
            'job_id': 'Job ID'
        }

        cols_to_display = [col for col in display_columns.keys() if col in filtered_df.columns]

        if not cols_to_display:
            st.error("The history data is missing key columns to display.")
        else:
            display_df = filtered_df[cols_to_display].rename(columns=display_columns)
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'Test Accuracy (%)' in display_df.columns:
                display_df['Test Accuracy (%)'] = display_df['Test Accuracy (%)'].map('{:.2f}%'.format)

            st.dataframe(display_df, use_container_width=True)

            st.header("🔬 Inspect a Specific Run")

            job_ids = filtered_df['job_id'].tolist()
            selected_job_id = st.selectbox("Select a Job ID to see full details", options=job_ids)

        if selected_job_id:
            job_details_row = filtered_df[filtered_df['job_id'] == selected_job_id].iloc[0]

            if 'params' in job_details_row and pd.notna(job_details_row['params']):
                model_type = job_details_row.get('model_type', 'model')
                date_str = pd.to_datetime(job_details_row.get('created_at')).strftime('%Y%m%d_%H%M%S')
                file_name = f"{date_str}_{model_type}_params.json"

                params_data = job_details_row['params']
                if not isinstance(params_data, str):
                    params_data = json.dumps(params_data, indent=2)

                st.download_button(
                    label="📥 Download Model Params",
                    data=params_data.encode('utf-8'),
                    file_name=file_name,
                    mime="application/json"
                )
            else:
                st.info("No downloadable parameters for this run.")

            st.subheader("💾 Save Model to Database")

            if 'params' not in job_details_row or not pd.notna(job_details_row['params']):
                st.info("Model can only be saved if the job is completed.")
            else:
                model_name_input = st.text_input("Model Name", key="model_name_input")

                if st.button("✅ Save Model"):
                    if not model_name_input.strip():
                        st.error("Please enter a model name before saving.")
                    else:
                        try:
                            response = client.save_model(
                                job_id=selected_job_id,
                                name=model_name_input.strip()
                            )

                            if response.get("status") == "created":
                                st.success("Model saved successfully! 🎉")
                                st.balloons()
                            elif response.get("status") == "exists":
                                st.warning("A model with this name already exists.")
                            elif response.get("status") == "not_found":
                                st.error("The associated training job was not found.")
                            elif response.get("status") == "error":
                                st.error(f"Error saving model: {response.get('message')}")
                            else:
                                st.error(f"Unexpected response: {response}")
                        except Exception as e:
                            st.error(f"Failed to save model: {e}")


            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⚙️ Configuration")
                config_data = {
                    k.replace('config.', ''): v
                    for k, v in job_details_row.to_dict().items()
                    if 'config.' in k
                }
                st.json(config_data)
            with col2:
                st.subheader("📈 Metrics")
                metrics_data = {
                    k.replace('metrics.', ''): v
                    for k, v in job_details_row.to_dict().items()
                    if 'metrics.' in k
                }
                st.json(metrics_data)


except Exception as e:
    st.error(f"Failed to fetch or display history: {e}")