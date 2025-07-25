import mini_keras as mk
import numpy as np

from pyrust.src.api.service.trainers.base import BaseTrainer


class MLPTrainer(BaseTrainer):
    def _prepare_config(self, config):
        experiment_config = super()._prepare_config(config)
        experiment_config.update(
            {
                "learning_rate": config.get("learning_rate", 0.01),
                "epochs": config.get("epochs", 1000),
                "layers": config.get("hidden_layer_sizes", [2, 2]),
                "activations": config.get("activations", ["linear"] * 2),
                "threshold": config.get("threshold", 0.5),
                "real_label": 1.0,
                "ai_label": 0.0,
            }
        )
        return experiment_config

    def _train_model(self, data):
        self.model = mk.MLP(
            is_classification=True,
            layers=self.experiment_config["layers"],
            activations=self.experiment_config["activations"],
        )
        data["y_train"] = np.array(data["y_train"]).reshape(-1, 1).tolist()
        data["y_test"] = np.array(data["y_test"]).reshape(-1, 1).tolist()
        self.model.fit(
            x_train=data["X_train"],
            y_train=data["y_train"],
            x_test=data["X_test"],
            y_test=data["y_test"],
            lr=self.experiment_config["learning_rate"],
            epochs=self.experiment_config["epochs"],
        )

    def _evaluate_model(self, data):
        threshold = self.experiment_config["threshold"]

        y_train_flat = np.ravel(data["y_train"])
        y_test_flat = np.ravel(data["y_test"])

        train_preds = [
            (0 if self.model.predict(x)[0] < threshold else 1) for x in data["X_train"]
        ]
        train_correct = sum(int(a == b) for a, b in zip(y_train_flat, train_preds))
        train_accuracy = (
            (train_correct / len(y_train_flat) * 100) if y_train_flat.size > 0 else 0
        )

        test_preds = [
            (0 if self.model.predict(x)[0] < threshold else 1) for x in data["X_test"]
        ]
        test_correct = sum(int(a == b) for a, b in zip(y_test_flat, test_preds))
        test_accuracy = (
            (test_correct / len(y_test_flat) * 100) if y_test_flat.size > 0 else 0
        )

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_samples": len(data["X_train"]),
            "test_samples": len(data["X_test"]),
            "len_real_images": data["loaded_counts"]["real"],
            "len_ai_images": data["loaded_counts"]["ai"],
            "total_images": len(data["X_train"]) + len(data["X_test"]),
            "train_losses": self.model.train_losses,
            "test_losses": self.model.test_losses,
            "train_accuracies": self.model.train_accuracies,
            "test_accuracies": self.model.test_accuracies,
        }
