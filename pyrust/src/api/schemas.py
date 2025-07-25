from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Tuple, Dict, Any
from typing import Optional


# Training
class ImageClassificationParams(BaseModel):
    max_images_per_class: int = 90
    image_height: int = 32
    image_width: int = 32


class LinearClassificationParams(ImageClassificationParams):
    learning_rate: float = 0.01
    max_iterations: int = 1000
    verbose: bool = True


class SVMParams(ImageClassificationParams):
    C: float = 1.0
    kernel: str = "rbf"
    gamma: Optional[float] = None


class MLPParams(ImageClassificationParams):
    learning_rate: float = 0.01
    hidden_layer_sizes: List[int] = [2, 2]
    activations: List[str] = ["linear", "linear"]
    epochs: int = 1000

class RBFParams(ImageClassificationParams):
    k: Optional[int] = None
    gamma: float = 0.01
    max_iterations: Optional[int] = None

# ----------------------------------
# History

from bson import ObjectId
from pydantic_core import core_schema


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """
        Définit la logique de validation. Remplace __get_validators__.
        """

        def validate(v):
            if not ObjectId.is_valid(v):
                raise ValueError("Invalid ObjectId")
            return ObjectId(v)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.no_info_plain_validator_function(validate),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}

class TrainingJob(BaseModel):
    id: PyObjectId = Field(alias="_id")
    job_id: str
    model_type: str
    status: str
    image_config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    model_saved: bool
    metrics: Dict[str, Any] = None
    created_at: datetime
    started_at: datetime = None
    finished_at: datetime = None
    params_file: str | None = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )
