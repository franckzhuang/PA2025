from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from pyrust.src.api.service.evaluators.linear import evaluate_linear
from pyrust.src.api.service.evaluators.mlp import evaluate_mlp
from pyrust.src.api.service.evaluators.svm import evaluate_svm

router = APIRouter()
 
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(HERE, "models")

class EvaluateRequest(BaseModel):
    model_type: str
    model_name: str
    input_data: list[float]


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
