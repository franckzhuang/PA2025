from enum import Enum


class ModelType(Enum):
    SVM = "SVM"
    LINEAR = "LINEAR"
    MLP = "MLP"
    RBF = "RBF"


class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"
