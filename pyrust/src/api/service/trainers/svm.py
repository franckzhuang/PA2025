import mini_keras as mk
from pyrust.src.api.service.trainers.base import BaseTrainer


class SVMTrainer(BaseTrainer):
    def _prepare_config(self, config):
        experiment_config = super()._prepare_config(config)
        kernel = config.get("kernel", "linear")
        gamma = config.get("gamma")
        threshold = config.get("threshold")
        if threshold is None:
            threshold = 0.0

        experiment_config.update(
            {
                "C": config.get("C", 1.0),
                "kernel": kernel,
                "gamma": gamma,
                "threshold": threshold,
                "real_label": 1.0,
                "ai_label": -1.0,
            }
        )

        return experiment_config

    def _train_model(self, data):
        self.model = mk.SVM(
            c=self.experiment_config["C"],
            kernel=self.experiment_config["kernel"],
            gamma=self.experiment_config["gamma"],
        )
        self.model.fit(data["X_train"], data["y_train"])

    def _evaluate_model(self, data):
        threshold = self.experiment_config["threshold"]

        raw_train_preds = self.model.predict(data["X_train"])
        train_preds = [(-1 if p < threshold else 1) for p in raw_train_preds]
        train_correct = sum(int(a == b) for a, b in zip(data["y_train"], train_preds))
        train_accuracy = (
            (train_correct / len(data["y_train"]) * 100) if data["y_train"] else 0
        )

        raw_test_preds = self.model.predict(data["X_test"])
        test_preds = [(-1 if p < threshold else 1) for p in raw_test_preds]
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
