from pathlib import Path
import mini_keras as mk
from datetime import datetime, timezone

from pyrust.src.api.models import Status
from pyrust.src.utils import DataLoader


def train_linear_classification(config, mongo_collection, job_id):
    base_path = Path(__file__).parent.parent.parent

    experiment_config = {
        "image_size": (32, 32),
        "max_images_per_class": config.get("max_images_per_class", 90),
        "real_images_path": config.get(
            "real_images_path", str(base_path / "data/real")
        ),
        "ai_images_path": config.get("ai_images_path", str(base_path / "data/ai")),
        "verbose": config.get("verbose", True),
        "learning_rate": config.get("learning_rate", 0.01),
        "max_iterations": config.get("max_iterations", 1000),
    }

    try:
        config_to_save = experiment_config.copy()
        config_to_save.pop("real_images_path", None)
        config_to_save.pop("ai_images_path", None)
        print(
            f"[{datetime.now(timezone.utc)}][{job_id}] Status: RUNNING | Config: {experiment_config}"
        )
        mongo_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": Status.RUNNING.value,
                    "started_at": datetime.now(timezone.utc),
                    "config": config_to_save,
                }
            },
            upsert=True,
        )

        print(f"[{datetime.now(timezone.utc)}][{job_id}] Loading dataset...")
        data = DataLoader.load_data(experiment_config)
        total_images = len(data["X_train"]) + len(data["X_test"])
        len_real = data["loaded_counts"]["real"]
        len_ai = data["loaded_counts"]["ai"]

        if total_images < 2:
            print(
                f"[{datetime.now(timezone.utc)}][{job_id}] FAILURE: Not enough images loaded (real: {len_real}, ai: {len_ai})"
            )
            mongo_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": Status.FAILURE.value,
                        "error": "Not enough images loaded",
                        "metrics": None,
                        "finished_at": datetime.now(timezone.utc),
                    }
                },
            )
            return {
                "status": Status.FAILURE.value,
                "error": "Not enough images loaded",
                "metrics": None,
                "config": experiment_config,
            }

        print(
            f"[{datetime.now(timezone.utc)}][{job_id}] Starting training ({len(data['X_train'])} samples)..."
        )
        model = mk.LinearClassification(
            verbose=experiment_config["verbose"],
            learning_rate=experiment_config["learning_rate"],
            max_iterations=experiment_config["max_iterations"],
        )
        model.fit(data["X_train"], data["y_train"])

        print(
            f"[{datetime.now(timezone.utc)}][{job_id}] Training finished. Evaluating..."
        )
        train_preds = [model.predict(x) for x in data["X_train"]]
        train_correct = sum(int(a == b) for a, b in zip(data["y_train"], train_preds))
        train_accuracy = (
            (train_correct / len(data["y_train"])) * 100 if data["y_train"] else 0
        )

        test_preds = [model.predict(x) for x in data["X_test"]]
        test_correct = sum(int(a == b) for a, b in zip(data["y_test"], test_preds))
        test_accuracy = (
            (test_correct / len(data["y_test"])) * 100 if data["y_test"] else 0
        )

        model_params = model.to_json()

        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_samples": len(data["X_train"]),
            "test_samples": len(data["X_test"]),
            "len_real_images": len_real,
            "len_ai_images": len_ai,
            "total_images": total_images,
        }

        print(
            f"[{datetime.now(timezone.utc)}][{job_id}] Success: Train acc={train_accuracy:.2f}%, Test acc={test_accuracy:.2f}%"
        )
        mongo_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": Status.SUCCESS.value,
                    "metrics": metrics,
                    "finished_at": datetime.now(timezone.utc),
                    "params": model_params,
                }
            },
        )

        return {
            "status": Status.SUCCESS.value,
            "metrics": metrics,
            "config": experiment_config,
        }

    except Exception as e:
        print(f"[{datetime.now(timezone.utc)}][{job_id}] ERROR: {str(e)}")
        mongo_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": Status.FAILURE.value,
                    "error": str(e),
                    "finished_at": datetime.now(timezone.utc),
                }
            },
        )
        return {
            "status": Status.FAILURE.value,
            "error": str(e),
            "metrics": None,
            "config": experiment_config,
        }
