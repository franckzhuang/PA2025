from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from pymongo.collection import Collection
import os
from pymongo.errors import PyMongoError

from pyrust.src.api.service.evaluators.linear import evaluate_linear
from pyrust.src.api.service.evaluators.mlp import evaluate_mlp
from pyrust.src.api.service.evaluators.svm import evaluate_svm
from pyrust.src.database.mongo import MongoDB

router = APIRouter()
 
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(HERE, "models")

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

@router.get("/evaluate/models")
def get_all_models():
    if not os.path.exists(MODELS_FOLDER):
        print("Models folder does not exist.")
        return {}

    model_types = {
        "linear": "LinearClassification",
        "mlp": "MLP",
        "svm": "SVM"
    }

    models = {v: [] for v in model_types.values()}

    for filename in os.listdir(MODELS_FOLDER):
        if filename.endswith(".json"):
            lower_name = filename.lower()
            for key, type_name in model_types.items():
                if key in lower_name:
                    name = os.path.splitext(filename)[0]
                    models[type_name].append(name)
                    break

    return {k: v for k, v in models.items() if v}


@router.post("/evaluate/run")
def evaluate_model(req: EvaluateRequest):
    model_path = os.path.join(MODELS_FOLDER, f"{req.model_name}.json")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")

    with open(model_path, "r") as f:
        json_str = f.read()

    if req.model_type == "LinearClassification":
        return evaluate_linear(json_str, req.input_data)

    if req.model_type == "MLP":
        return evaluate_mlp(json_str, req.input_data)

    if req.model_type == "SVM":
        return evaluate_svm(json_str, req.input_data)

    raise HTTPException(status_code=400, detail="Invalid model type.")

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

    try:
        result = collection.insert_one({
            "name": request.name,
            "job_id": request.job_id
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