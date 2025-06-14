from datetime import datetime, timezone

from fastapi import FastAPI, BackgroundTasks, Depends
from uuid import uuid4
from datetime import datetime

from pyrust.src.api.models import Status
from pyrust.src.api.schemas import (
    LinearClassificationParams,
    SVMParams,
    MLPParams,
    KMeansParams,
)
from pyrust.src.api.service.linear_classification import train_linear_classification

# from service.svm import train_svm
# from service.mlp import train_mlp
# from service.kmeans import train_kmeans
from pyrust.src.database.mongo import MongoDB

app = FastAPI()


def get_mongo_collection():
    mongo = MongoDB()
    return mongo.db["training_jobs"]


@app.post("/train/linear_classification")
def train_linear_classification_controller(
    params: LinearClassificationParams,
    background_tasks: BackgroundTasks,
    collection=Depends(get_mongo_collection),
):
    job_id = str(uuid4())
    collection.insert_one(
        {
            "job_id": job_id,
            "model_type": "linear_classification",
            "status": Status.RUNNING.value,
            "created_at": datetime.now(timezone.utc),
        }
    )
    background_tasks.add_task(
        run_training_job, job_id, "linear", params.dict(), collection
    )
    return {"job_id": job_id}


# @app.post("/train/svm")
# def train_svm_controller(params: SVMParams, background_tasks: BackgroundTasks, collection=Depends(get_mongo_collection)):
#     job_id = str(uuid4())
#     collection.insert_one({
#         "job_id": job_id,
#         "model_type": "svm",
#         "status": "pending",
#         "created_at": datetime.utcnow()
#     })
#     background_tasks.add_task(run_training_job, job_id, "svm", params.dict(), collection)
#     return {"job_id": job_id}


@app.get("/train/status/{job_id}")
def train_status(job_id: str, collection=Depends(get_mongo_collection)):
    job = collection.find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        return {"status": "not_found"}
    return job


def run_training_job(job_id, model_type: str, config, collection):
    try:
        collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": Status.RUNNING.value,
                    "started_at": datetime.now(timezone.utc),
                }
            },
        )
        if model_type == "linear":
            train_linear_classification(config, collection, job_id)
        # elif model_type == "svm":
        #     train_svm(config, collection, job_id)
        # elif model_type == "mlp":
        #     train_mlp(config, collection, job_id)
        # elif model_type == "kmeans":
        #     train_kmeans(config, collection, job_id)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": Status.FAILURE.value,
                    "error": str(e),
                    "finished_at": datetime.now(timezone.utc),
                }
            },
        )
