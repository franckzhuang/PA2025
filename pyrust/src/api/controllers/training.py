from typing import List

from fastapi import BackgroundTasks, Depends, APIRouter
from uuid import uuid4
from datetime import datetime, timezone, timedelta
import json

from pyrust.src.api.models import Status, ModelType
from pyrust.src.api.schemas import (
    LinearClassificationParams,
    MLPParams,
    SVMParams,
    TrainingJob,
)
from pyrust.src.api.service.trainers.svm import SVMTrainer
from pyrust.src.database.mongo import MongoDB
from pyrust.src.api.service.trainers.linear import LinearClassificationTrainer
from pyrust.src.api.service.trainers.mlp import MLPTrainer

router = APIRouter()


def get_mongo_collection():
    mongo = MongoDB()
    return mongo.db["training_jobs"]

def get_saved_models_collection():
    mongo = MongoDB()
    return mongo.db["saved_models"]


def run_training_job(trainer_class, config, collection, job_id):
    trainer = trainer_class(config, collection, job_id)
    trainer.run()


def cleanup_old_model_params(collection):
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    collection.update_many(
        {"created_at": {"$lt": one_hour_ago}}, {"$unset": {"params": ""}}
    )


@router.post("/train/linear_classification", status_code=202)
def train_linear_classification(
    params: LinearClassificationParams,
    background_tasks: BackgroundTasks,
    collection=Depends(get_mongo_collection),
):
    cleanup_old_model_params(collection)
    job_id = str(uuid4())
    collection.insert_one(
        {
            "job_id": job_id,
            "model_type": ModelType.LINEAR.value,
            "status": Status.RUNNING.value,
            "created_at": datetime.now(timezone.utc),
        }
    )
    background_tasks.add_task(
        run_training_job,
        LinearClassificationTrainer,
        params.model_dump(),
        collection,
        job_id,
    )
    return {"job_id": job_id}


@router.post("/train/mlp", status_code=202)
def train_mlp(
    params: MLPParams,
    background_tasks: BackgroundTasks,
    collection=Depends(get_mongo_collection),
):
    cleanup_old_model_params(collection)
    job_id = str(uuid4())
    collection.insert_one(
        {
            "job_id": job_id,
            "model_type": ModelType.MLP.value,
            "status": Status.RUNNING.value,
            "created_at": datetime.now(timezone.utc),
        }
    )
    background_tasks.add_task(
        run_training_job, MLPTrainer, params.model_dump(), collection, job_id
    )
    return {"job_id": job_id}


@router.post("/train/svm", status_code=202)
def train_svm(
    params: SVMParams,
    background_tasks: BackgroundTasks,
    collection=Depends(get_mongo_collection),
):
    cleanup_old_model_params(collection)
    job_id = str(uuid4())
    collection.insert_one(
        {
            "job_id": job_id,
            "model_type": ModelType.SVM.value,
            "status": Status.RUNNING.value,
            "created_at": datetime.now(timezone.utc),
        }
    )
    background_tasks.add_task(
        run_training_job, SVMTrainer, params.model_dump(), collection, job_id
    )
    return {"job_id": job_id}


@router.post("/train/rbf", status_code=202)
def train_svm(
    params: SVMParams,
    background_tasks: BackgroundTasks,
    collection=Depends(get_mongo_collection),
):
    cleanup_old_model_params(collection)
    job_id = str(uuid4())
    collection.insert_one(
        {
            "job_id": job_id,
            "model_type": ModelType.RBF.value,
            "status": Status.RUNNING.value,
            "created_at": datetime.now(timezone.utc),
        }
    )
    background_tasks.add_task(
        run_training_job, SVMTrainer, params.model_dump(), collection, job_id
    )
    return {"job_id": job_id}


@router.get("/train/status/{job_id}")
def get_training_status(job_id: str, collection=Depends(get_mongo_collection)):
    job = collection.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        return {"status": "not_found"}
    return job


@router.get(
    "/train",
    response_model=List[TrainingJob],
    summary="Get Training History",
    description="Fetches the last 100 training job records, sorted by creation date.",
)
def get_history(collection=Depends(get_mongo_collection)):
    history_cursor = collection.find().sort("created_at", -1).limit(100)
    history_list = history_cursor.to_list(length=100)
    return history_list

@router.post("/training/import_model")
def import_model(
    data: dict,
    training_jobs=Depends(get_mongo_collection),
    saved_models=Depends(get_saved_models_collection)
):
    job = data.get("job")
    if not job:
        return {"status": "error", "message": "Missing 'job' section in payload."}

    job_id = job.get("job_id")
    if not job_id:
        return {"status": "error", "message": "Missing 'job_id' in job section."}

    if training_jobs.find_one({"job_id": job_id}):
        return {"status": "exists", "message": "A training job with this job_id already exists."}
    
    raw_params = data.get("params")
    if isinstance(raw_params, (dict, list)):
        params_str = json.dumps(raw_params, indent=2)
    else:
        params_str = str(raw_params)

    training_doc = {
        "job_id": job_id,
        "model_type": job.get("model_type", "UNKNOWN").upper(),
        "status": job.get("status", "SUCCESS"),
        "params": params_str,
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "hyperparameters": job.get("hyperparameters"),
        "image_config": job.get("image_config"),
        "metrics": job.get("metrics")
    }
    training_jobs.insert_one(training_doc)

    saved_model_doc = {
        "name": f"Imported {job.get('model_type', 'Model')} - {job.get('created_at')}",
        "job_id": job_id,
        "model_type": job.get("model_type", "UNKNOWN").upper(),
        "created_at": job.get("created_at")
    }
    saved_models.insert_one(saved_model_doc)

    return {
        "status": "created",
        "job_id": job_id,
        "model_name": saved_model_doc["name"]
    }
