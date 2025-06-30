import mini_keras as mk
from pyrust.src.api.service.trainers.base import BaseTrainer


class MLPTrainer(BaseTrainer):
    def _prepare_config(self, config):
        return {
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
            "learning_rate": config.get("learning_rate", 0.01),
            "epochs": config.get("epochs", 1000),
            "layers": config.get("hidden_layer_sizes", [2, 2]),
            "threshold": config.get("threshold", 0.5),
        }

    def _train_model(self, data):
        self.model = mk.MLP(
            is_classification=True,
            layers=self.experiment_config["layers"],
        )
        self.model.fit(
            data["X_train"],
            data["y_train"],
            lr=self.experiment_config["learning_rate"],
            epochs=self.experiment_config["epochs"],
        )

    def _evaluate_model(self, data):
        threshold = self.experiment_config["threshold"]

        train_preds = [
            (-1 if self.model.predict(x)[0] < threshold else 1) for x in data["X_train"]
        ]
        train_correct = sum(int(a == b) for a, b in zip(data["y_train"], train_preds))
        train_accuracy = (
            (train_correct / len(data["y_train"]) * 100) if data["y_train"] else 0
        )

        test_preds = [
            (-1 if self.model.predict(x)[0] < threshold else 1) for x in data["X_test"]
        ]
        test_correct = sum(int(a == b) for a, b in zip(data["y_test"], test_preds))
        test_accuracy = (
            (test_correct / len(data["y_test"]) * 100) if data["y_test"] else 0
        )

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_samples": len(data["X_train"]),
            "test_samples": len(data["X_test"]),
            "len_real_images": data["loaded_counts"]["real"],
            "len_ai_images": data["loaded_counts"]["ai"],
            "total_images": len(data["X_train"]) + len(data["X_test"]),
        }
