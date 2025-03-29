FROM python:3.12-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

ENV PYTHONUNBUFFERED=1
ENV UV_PYTHON=/usr/local/bin/python

WORKDIR /app

COPY requirements.txt .

RUN uv pip install -r requirements.txt

RUN playwright install chromium && \
    playwright install-deps

COPY src/ .


CMD ["python", "main.py"]