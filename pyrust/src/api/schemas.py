from pydantic import BaseModel
from typing import List, Optional


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


class MLPParams(ImageClassificationParams):
    learning_rate: float = 0.01
    hidden_layer_sizes: List[int] = [2, 2]
    epochs: int = 1000


class KMeansParams(ImageClassificationParams):
    n_clusters: int = 2
    max_iterations: int = 300
