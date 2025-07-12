from typing import List

from fastapi import BackgroundTasks, Depends, APIRouter
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from pyrust.src.api.models import Status, ModelType
from pyrust.src.api.schemas import (
    LinearClassificationParams,
    MLPParams, SVMParams, TrainingHistory,
)
from pyrust.src.api.service.trainers.svm import SVMTrainer
from pyrust.src.database.mongo import MongoDB
from pyrust.src.api.service.trainers.linear import LinearClassificationTrainer
from pyrust.src.api.service.trainers.mlp import MLPTrainer

router = APIRouter()


def get_mongo_collection():
    mongo = MongoDB()
    return mongo.db["training_jobs"]


def run_training_job(trainer_class, config, collection, job_id):
    trainer = trainer_class(config, collection, job_id)
    trainer.run()

def cleanup_old_model_params(collection):
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    collection.update_many(
        {"created_at": {"$lt": one_hour_ago}},
        {"$unset": {"params": ""}}
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


@router.get("/train/status/{job_id}")
def get_training_status(job_id: str, collection=Depends(get_mongo_collection)):
    job = collection.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        return {"status": "not_found"}
    return job


@router.get(
    "/train/history",
    response_model=List[TrainingHistory],
    summary="Get Training History",
    description="Fetches the last 100 training job records, sorted by creation date.",
)
def get_history(collection=Depends(get_mongo_collection)):
    history_cursor = collection.find().sort("created_at", -1).limit(100)
    history_list = history_cursor.to_list(length=100)
    return history_list