from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timezone
import logging
from pyrust.src.api.models import Status
from pyrust.src.utils.data_loader import DataLoader
from pyrust.src.utils.logger import logger, log_with_job_id


class BaseTrainer(ABC):
    def __init__(self, config, collection, job_id):
        self.job_id = job_id
        self.collection = collection
        self.base_path = Path(__file__).parent.parent.parent.parent
        self.experiment_config = self._prepare_config(config)
        self.model = None

    @abstractmethod
    def _prepare_config(self, config):
        pass

    @abstractmethod
    def _train_model(self, data):
        pass

    @abstractmethod
    def _evaluate_model(self, data):
        pass

    def run(self):
        try:
            self._update_status(Status.RUNNING, {"config": self._get_savable_config()})
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

            log_with_job_id(
                logger,
                self.job_id,
                f"Starting training ({len(data['X_train'])} samples)...",
            )
            self._train_model(data)

            log_with_job_id(logger, self.job_id, "Training finished. Evaluating...")
            metrics = self._evaluate_model(data)

            model_params = (
                self.model.to_json() if hasattr(self.model, "to_json") else None
            )

            update_data = {"metrics": metrics, "params": model_params}
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
        """Loads the dataset."""
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
        """Updates the job status in the database."""
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

    def _get_savable_config(self):
        cfg = self.experiment_config.copy()
        cfg.pop("real_images_path", None)
        cfg.pop("ai_images_path", None)
        return cfg

    def _build_response(self, status, metrics=None, error=None):
        return {
            "status": status.value,
            "metrics": metrics,
            "error": error,
            "config": self.experiment_config,
        }
