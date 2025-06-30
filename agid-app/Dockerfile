FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y git libcurl4 libssl3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install streamlit dotenv plotly matplotlib

WORKDIR /app

COPY src/ .

CMD ["streamlit", "run", "Detector.py"]