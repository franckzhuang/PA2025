from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from pymongo.collection import Collection
import os
from datetime import datetime
from pymongo.errors import PyMongoError

from pyrust.src.api.service.evaluators.linear import evaluate_linear
from pyrust.src.api.service.evaluators.mlp import evaluate_mlp
from pyrust.src.api.service.evaluators.svm import evaluate_svm
from pyrust.src.database.mongo import MongoDB

router = APIRouter()

class EvaluateRequest(BaseModel):
    model_type: str
    model_name: str
    input_data: list[float]

class SaveModelRequest(BaseModel):
    job_id: str
    name: str

def get_saved_models_collection() -> Collection:
    mongo = MongoDB()
    return mongo.db["saved_models"]

def get_training_jobs_collection():
    mongo = MongoDB()
    return mongo.db["training_jobs"]

@router.get("/evaluate/models")
def get_all_models(collection=Depends(get_saved_models_collection)):
    models_cursor = collection.find()
    models = {}
    for doc in models_cursor:
        mtype = doc.get("model_type")
        if mtype not in models:
            models[mtype] = []
        models[mtype].append(doc["name"])
    return models


@router.post("/evaluate/run")
def evaluate_model(
    req: EvaluateRequest,
    saved_models=Depends(get_saved_models_collection),
    training_jobs=Depends(get_training_jobs_collection)
):
    model_doc = saved_models.find_one({"name": req.model_name})
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found in saved_models.")
    
    stored_type = model_doc.get("model_type")
    if not stored_type:
        raise HTTPException(status_code=400, detail="Model type not stored in saved_models.")
    
    if stored_type != req.model_type:
        raise HTTPException(
            status_code=400,
            detail=f"Model type mismatch. Requested: {req.model_type}, Stored: {stored_type}"
        )

    job_doc = training_jobs.find_one({"job_id": model_doc["job_id"]})
    if not job_doc:
        raise HTTPException(status_code=404, detail="Training job not found.")

    params = job_doc.get("params")
    if not params:
        raise HTTPException(status_code=404, detail="No params found for this job.")

    if stored_type == "LINEAR":
        return evaluate_linear(params, req.input_data)
    elif stored_type == "MLP":
        return evaluate_mlp(params, req.input_data)
    elif stored_type == "SVM":
        return evaluate_svm(params, req.input_data)

    raise HTTPException(status_code=400, detail=f"Invalid model type: {stored_type}")

@router.get("/evaluate/saved_models")
def get_saved_models(collection=Depends(get_saved_models_collection)):
    docs = list(collection.find({}, {"_id": 1, "name": 1, "job_id": 1}))
    return [
        {
            "id": str(doc["_id"]),
            "name": doc["name"],
            "job_id": doc["job_id"]
        }
        for doc in docs
    ]


@router.post("/evaluate/save_model")
def save_model(request: SaveModelRequest):
    mongo = MongoDB()
    collection = mongo.db["saved_models"]

    if collection.find_one({"name": request.name}):
        return {
            "status": "exists",
            "message": "A model with this name already exists."
        }

    job_doc = mongo.db["training_jobs"].find_one({"job_id": request.job_id})
    if not job_doc:
        return {
            "status": "not_found",
            "message": "Training job not found."
        }

    model_type = job_doc.get("model_type", "Unknown")

    try:
        result = collection.insert_one({
            "name": request.name,
            "job_id": request.job_id,
            "model_type": model_type,
            "created_at": datetime.utcnow()
        })
        return {
            "status": "created",
            "id": str(result.inserted_id)
        }
    except PyMongoError as e:
        return {
            "status": "error",
            "message": f"Failed to save model: {e}"
        }

@router.get("/models/details/{model_name}")
def get_model_details(
    model_name: str,
    saved_models=Depends(get_saved_models_collection),
    training_jobs=Depends(get_training_jobs_collection)
):
    try:
        model_doc = saved_models.find_one({"name": model_name})
        if not model_doc:
            raise HTTPException(status_code=404, detail="Model not found.")

        job_id = model_doc.get("job_id")
        job_doc = training_jobs.find_one({"job_id": job_id})

        if not job_doc:
            raise HTTPException(status_code=404, detail="Training job not found.")

        model_info = {
            "model_name": model_doc.get("name"),
            "model_type": model_doc.get("model_type"),
            "created_at": model_doc.get("created_at").isoformat() if model_doc.get("created_at") else None,
            "job": {
                "job_id": job_doc.get("job_id"),
                "config": job_doc.get("config"),
                "params": job_doc.get("params"),
                "created_at": job_doc.get("created_at").isoformat() if job_doc.get("created_at") else None
            }
        }
        return model_info

    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")