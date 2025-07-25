import mini_keras as mk
from pyrust.src.api.service.trainers.base import BaseTrainer


class RBFTrainer(BaseTrainer):
    def _prepare_config(self, config):
        experiment_config = super()._prepare_config(config)
        k = config.get("k")
        gamma = config.get("gamma", 0.01)
        max_iterations = config.get("max_iterations")

        if k and max_iterations:
            experiment_config.update({
                "k": k,
                "max_iterations": max_iterations,
                "gamma": gamma,
                "real_label": 1.0,
                "ai_label": -1.0,
                "threshold": config.get("threshold", 0.51),
            })
        else:
            experiment_config.update({
                "gamma": gamma,
                "real_label": 1.0,
                "ai_label": -1.0,
                "threshold": config.get("threshold", 0.51),
            })

        return experiment_config

    def _train_model(self, data):
        if self.experiment_config.get("k") and self.experiment_config.get("max_iterations"):
            self.model = mk.RBFKMeans(
                x=data["X_train"],
                y=data["y_train"],
                k=self.experiment_config["k"],
                max_iters=self.experiment_config["max_iterations"],
                gamma=self.experiment_config["gamma"],
                is_classification=True,
            )
        else:
            self.model = mk.RBFNaive(
                x=data["X_train"],
                y=data["y_train"],
                gamma=self.experiment_config["gamma"],
                is_classification=True,
            )

    def _evaluate_model(self, data):
        threshold = self.experiment_config["threshold"]


        train_preds = [
            (-1 if self.model.predict(x) < threshold else 1) for x in data["X_train"]

        ]
        train_correct = sum(int(a == b) for a, b in zip(data["y_train"], train_preds))
        train_accuracy = (train_correct / len(data["y_train"]) * 100)


        test_preds = [
            (-1 if self.model.predict(x) < threshold else 1) for x in data["X_test"]
        ]
        test_correct = sum(int(a == b) for a, b in zip(data["y_test"], test_preds))
        test_accuracy = (test_correct / len(data["y_test"]) * 100)


        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_samples": len(data["X_train"]),
            "test_samples": len(data["X_test"]),
            "len_real_images": data["loaded_counts"]["real"],
            "len_ai_images": data["loaded_counts"]["ai"],
            "total_images": len(data["X_train"]) + len(data["X_test"]),
        }
