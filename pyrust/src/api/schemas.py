# schemas.py

from pydantic import BaseModel
from typing import List

class LinearClassificationParams(BaseModel):
    max_images_per_class: int
    learning_rate: float
    max_iterations: int

class SVMParams(BaseModel):
    max_images_per_class: int
    C: float
    kernel: str

class MLPParams(BaseModel):
    max_images_per_class: int
    learning_rate: float
    hidden_layer_sizes: List[int]
    max_iterations: int

class KMeansParams(BaseModel):
    max_images_per_class: int
    n_clusters: int
    max_iterations: int
