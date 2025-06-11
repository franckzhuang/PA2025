# main.py

from fastapi import FastAPI, BackgroundTasks
from uuid import uuid4

from pyrust.src.api.models import ModelType
from schemas import LinearClassificationParams, SVMParams, MLPParams, KMeansParams
from service.linear_classification import train_linear_classification
from service.svm import train_svm
from service.mlp import train_mlp
from service.kmeans import train_kmeans

app = FastAPI()
training_jobs = {}

# Linear Classification
@app.post("/train/linear_classification")
def train_linear_classification_controller(params: LinearClassificationParams, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    training_jobs[job_id] = {"status": "pending", "metrics": None, "error": None}
    background_tasks.add_task(run_training_job, job_id, "linear_classification", params.dict())
    return {"job_id": job_id}

# SVM
@app.post("/train/svm")
def train_svm_controller(params: SVMParams, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    training_jobs[job_id] = {"status": "pending", "metrics": None, "error": None}
    background_tasks.add_task(run_training_job, job_id, "svm", params.dict())
    return {"job_id": job_id}

# MLP
@app.post("/train/mlp")
def train_mlp_controller(params: MLPParams, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    training_jobs[job_id] = {"status": "pending", "metrics": None, "error": None}
    background_tasks.add_task(run_training_job, job_id, "mlp", params.dict())
    return {"job_id": job_id}

# KMeans
@app.post("/train/kmeans")
def train_kmeans_controller(params: KMeansParams, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    training_jobs[job_id] = {"status": "pending", "metrics": None, "error": None}
    background_tasks.add_task(run_training_job, job_id, "kmeans", params.dict())
    return {"job_id": job_id}

# Status endpoint
@app.get("/train/status/{job_id}")
def train_status(job_id: str):
    job = training_jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return job

def run_training_job(job_id, model_type: str, config):
    try:
        training_jobs[job_id]["status"] = "running"
        match model_type:
            case ModelType.LINEAR:
                metrics = train_linear_classification(config)
            case ModelType.SVM:
                metrics = train_svm(config)
            case ModelType.MLP:
                metrics = train_mlp(config)
            case ModelType.KMEANS:
                metrics = train_kmeans(config)
            case _:
                raise ValueError(f"Unknown model type: {model_type}")
        training_jobs[job_id]["status"] = "finished"
        training_jobs[job_id]["metrics"] = metrics
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
