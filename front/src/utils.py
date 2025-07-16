import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")


class ApiClient:
    """Client pour interagir avec l'API d'entraînement."""

    def __init__(self, base_url=API_URL, timeout=60, max_retries=3):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout
        adapter = requests.adapters.HTTPAdapter(max_retries=max_retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

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
        url = f"{self.base_url}/train"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_models(self):
        url = f"{self.base_url}/evaluate/models"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def evaluate_model(self, model_type, model_name, input_data):
        url = f"{self.base_url}/evaluate/run"
        payload = {
            "model_type": model_type,
            "model_name": model_name,
            "input_data": input_data,
        }
        resp = self.session.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def save_model(self, job_id, name):
        url = f"{self.base_url}/evaluate/save_model"
        payload = {"job_id": job_id, "name": name}
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_model_details(self, model_name: str):
        res = requests.get(f"{self.base_url}/models/details/{model_name}")
        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(f"Failed to get model details: {res.text}")

    def import_model(self, data: dict):
        url = f"{self.base_url}/training/import_model"
        resp = self.session.post(url, json=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

