import mini_keras as mk
from pyrust.src.api.service.trainers.base import BaseTrainer


class LinearClassificationTrainer(BaseTrainer):
    def _prepare_config(self, config):
        experiment_config = super()._prepare_config(config)

        experiment_config.update(
            {
                "verbose": config.get("verbose", True),
                "learning_rate": config.get("learning_rate", 0.01),
                "max_iterations": config.get("max_iterations", 1000),
                "real_label": 1.0,
                "ai_label": -1.0,
            }
        )
        return experiment_config

    def _train_model(self, data):
        self.model = mk.LinearClassification(
            verbose=self.experiment_config["verbose"],
            learning_rate=self.experiment_config["learning_rate"],
            max_iterations=self.experiment_config["max_iterations"],
        )
        self.model.fit(data["X_train"], data["y_train"])

    def _evaluate_model(self, data):
        train_preds = [self.model.predict(x) for x in data["X_train"]]
        train_correct = sum(int(a == b) for a, b in zip(data["y_train"], train_preds))
        train_accuracy = (
            (train_correct / len(data["y_train"]) * 100) if data["y_train"] else 0
        )

        test_preds = [self.model.predict(x) for x in data["X_test"]]
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
