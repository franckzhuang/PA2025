from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from pymongo.collection import Collection
from PIL import Image
import numpy as np
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
    input_data: list[list[list[float]]]

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

def get_image_size(
    model_name: str,
    saved_models=Depends(get_saved_models_collection),
    training_jobs=Depends(get_training_jobs_collection)
):
    model_doc = saved_models.find_one({"name": model_name})
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found.")

    job_id = model_doc.get("job_id")
    job_doc = training_jobs.find_one({"job_id": job_id})
    if not job_doc:
        raise HTTPException(status_code=404, detail="Training job not found.")

    image_config = job_doc.get("image_config")
    if not image_config or "image_size" not in image_config:
        raise HTTPException(status_code=404, detail="No image_size found for this model.")

    return image_config["image_size"]   


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
    
    img_size = get_image_size(
        model_name=req.model_name,
        saved_models=saved_models,
        training_jobs=training_jobs
    )
    raw_array = np.array(req.input_data, dtype=np.float32)
    img = Image.fromarray(raw_array.astype(np.uint8)).convert("RGB")
    img_resized = img.resize(tuple(img_size))
    input_data = (np.array(img_resized).astype(np.float32) / 255.0).flatten().tolist()


    job_doc = training_jobs.find_one({"job_id": model_doc["job_id"]})
    if not job_doc:
        raise HTTPException(status_code=404, detail="Training job not found.")

    params = job_doc.get("params")
    if not params:
        raise HTTPException(status_code=404, detail="No params found for this job.")

    if stored_type == "LINEAR":
        return evaluate_linear(params, input_data)
    elif stored_type == "MLP":
        return evaluate_mlp(params, input_data)
    elif stored_type == "SVM":
        return evaluate_svm(params, input_data)

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
    model_doc = saved_models.find_one({"name": model_name})
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found.")

    job_id = model_doc.get("job_id")
    job_doc = training_jobs.find_one({"job_id": job_id})
    if not job_doc:
        raise HTTPException(status_code=404, detail="Training job not found.")
    
        
    def safe_isoformat(value):
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        return None

    model_info = {
        "model_name": model_doc.get("name"),
        "model_type": model_doc.get("model_type"),
        "created_at": model_doc.get("created_at"),
        "job": {
            "job_id": job_doc.get("job_id"),
            "model_type": job_doc.get("model_type"),
            "status": job_doc.get("status"),
            "created_at": job_doc.get("created_at"),
            "started_at": job_doc.get("started_at"),
            "finished_at": job_doc.get("finished_at"),
            "hyperparameters": job_doc.get("hyperparameters"),
            "image_config": job_doc.get("image_config"),
            "metrics": job_doc.get("metrics"),
            "params": job_doc.get("params"),
        }
    }
    return model_info
