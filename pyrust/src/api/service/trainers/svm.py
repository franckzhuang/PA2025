import mini_keras as mk
from pyrust.src.api.service.trainers.base import BaseTrainer


class SVMTrainer(BaseTrainer):
    def _prepare_config(self, config):
        experiment_config = super()._prepare_config(config)
        experiment_config.update({
            "c": config.get("c"),
            "kernel": config.get("kernel", "linear"),
            "gamma": config.get("gamma"),
        })

    def _train_model(self, data):
        self.model = mk.SVM(
            c=self.experiment_config["c"],
            kernel=self.experiment_config["kernel"],
            gamma=self.experiment_config["gamma"],
        )
        self.model.fit(data["X_train"], data["y_train"])

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
