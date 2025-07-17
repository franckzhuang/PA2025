import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timezone
import logging
from pyrust.src.api.models import Status
from pyrust.src.utils.data_loader import DataLoader
from pyrust.src.utils.logger import logger, log_with_job_id


class BaseTrainer(ABC):
    def __init__(self, config, collection, job_id):
        try:
            self.job_id = job_id
            self.collection = collection

            self.base_path = Path(__file__).parent.parent.parent.parent
            self.models_path = self.base_path / "training_models"
            self.models_path.mkdir(parents=True, exist_ok=True)

            self.experiment_config = self._prepare_config(config)
            self.model = None
        except Exception as e:
            log_with_job_id(logger, job_id, f"Initialization error: {str(e)}", level=logging.ERROR)
            raise e

    def _prepare_config(self, config):
        base_config = {
            "image_size": (
                config.get("image_width", 32),
                config.get("image_height", 32),
            ),
            "max_images_per_class": config.get("max_images_per_class", 90),
            "real_images_path": config.get(
                "real_images_path", f"{str(self.base_path)}/data/real"
            ),
            "ai_images_path": config.get(
                "ai_images_path", f"{str(self.base_path)}/data/ai"
            ),
        }
        return base_config

    @abstractmethod
    def _train_model(self, data):
        pass

    @abstractmethod
    def _evaluate_model(self, data):
        pass

    def run(self):
        try:
            self._update_status(Status.RUNNING, self._get_savable_config())
            log_with_job_id(
                logger,
                self.job_id,
                f"Status: RUNNING | Config: {self.experiment_config}",
            )

            data = self._load_data()
            if not self._validate_data(data):
                return self._build_response(
                    Status.FAILURE, error="Not enough images loaded"
                )

            files = {
                "files_train": data["files_train"],
                "files_test": data["files_test"],
            }
            self._update_image_config(files)

            log_with_job_id(
                logger,
                self.job_id,
                f"Starting training ({len(data['X_train'])} samples)...",
            )
            training_start = time.perf_counter()
            self._train_model(data)
            training_end = time.perf_counter()
            training_duration = training_end - training_start

            log_with_job_id(logger, self.job_id, "Training finished. Evaluating...")
            metrics = self._evaluate_model(data)

            metrics["training_duration"] = training_duration

            model_params = (
                self.model.to_json() if hasattr(self.model, "to_json") else None
            )

            params_file = None
            if model_params:
                file_path = self.models_path / f"{self.job_id}_params.json"
                params_file = str(file_path)

                try:
                    with open(file_path, 'w') as f:
                        if isinstance(model_params, dict):
                            json.dump(model_params, f, indent=2)
                        else:
                            f.write(model_params)
                    log_with_job_id(logger, self.job_id, f"Model parameters saved to {params_file}")
                except Exception as e:
                    log_with_job_id(logger, self.job_id, f"Failed to save model params to file: {e}", level=logging.ERROR)
                    params_file = None

            update_data = {"metrics": metrics}
            if params_file:
                update_data["params_file"] = params_file.split("/")[-1]
                update_data["model_saved"] = False

            self._update_status(Status.SUCCESS, update_data)

            log_with_job_id(
                logger,
                self.job_id,
                f"Success: Train acc={metrics.get('train_accuracy', 0):.2f}%, Test acc={metrics.get('test_accuracy', 0):.2f}%",
            )
            return self._build_response(Status.SUCCESS, metrics=metrics)

        except Exception as e:
            log_with_job_id(
                logger, self.job_id, f"ERROR: {str(e)}", level=logging.ERROR
            )
            self._update_status(Status.FAILURE, {"error": str(e)})
            return self._build_response(Status.FAILURE, error=str(e))

    def _load_data(self):
        log_with_job_id(logger, self.job_id, "Loading dataset...")
        return DataLoader.load_data(self.experiment_config)

    def _validate_data(self, data):
        total_images = len(data["X_train"]) + len(data["X_test"])
        if total_images < 2:
            real_count = data["loaded_counts"]["real"]
            ai_count = data["loaded_counts"]["ai"]
            log_with_job_id(
                logger,
                self.job_id,
                f"FAILURE: Not enough images loaded (real: {real_count}, ai: {ai_count})",
                level=logging.ERROR,
            )
            return False
        return True

    def _update_status(self, status, additional_info=None):
        update_doc = {"status": status.value}
        if status == Status.RUNNING:
            update_doc["started_at"] = datetime.now(timezone.utc)
        else:
            update_doc["finished_at"] = datetime.now(timezone.utc)

        if additional_info:
            update_doc.update(additional_info)

        self.collection.update_one(
            {"job_id": self.job_id},
            {"$set": update_doc},
            upsert=(status == Status.RUNNING),  # Only upsert at start
        )

    def _update_image_config(self, image_config):
        if not image_config:
            return

        set_fields = {}
        for key, value in image_config.items():
            set_fields[f"image_config.{key}"] = value

        self.collection.update_one(
            {"job_id": self.job_id},
            {"$set": set_fields},
            upsert=True,
        )

    def _get_savable_config(self):
        cfg = self.experiment_config.copy()
        savable_config = {
            "image_config": {
                "image_size": cfg["image_size"],
                "images_per_class": cfg["max_images_per_class"],
            },
            "hyperparameters": cfg,
        }
        image_keys = [
            "real_images_path",
            "ai_images_path",
            "image_size",
            "max_images_per_class",
        ]
        for key in image_keys:
            savable_config["hyperparameters"].pop(key, None)
        return savable_config

    def _build_response(self, status, metrics=None, error=None):
        return {
            "status": status.value,
            "metrics": metrics,
            "error": error,
            "config": self.experiment_config,
        }
