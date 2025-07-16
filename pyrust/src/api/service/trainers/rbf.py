import mini_keras as mk
from pyrust.src.api.service.trainers.base import BaseTrainer


class RBFTrainer(BaseTrainer):
    def _prepare_config(self, config):
        experiment_config = super()._prepare_config(config)
        k = config.get("k")
        gamma = config.get("gamma", 0.01)
        max_iterations = config.get("max_iterations")


        experiment_config.update(
            {
                "k": k,
                "max_iterations": max_iterations,

            } if k and max_iterations else {
                "gamma": gamma,
            }
        )

        return experiment_config

    def _train_model(self, data):
        if self.experiment_config.get("gamma") and self.experiment_config.get("max_iterations"):
            self.model = mk.RBFKMeans(
                x=data["X_train"],
                y=data["y_train"],
                c=self.experiment_config["C"],
                kernel=self.experiment_config["kernel"],
                gamma=self.experiment_config["gamma"],
                is_classification=True,
            )
        else:
            self.model = mk.RBFNaive(
                x=data["X_train"],
                y=data["y_train"],
                k=self.experiment_config["k"],
                is_classification=True,
            )

    def _evaluate_model(self, data):
        raw_train_preds = self.model.predict(data["X_train"])
        train_preds = [(-1 if p == 0 else 1) for p in raw_train_preds]
        train_correct = sum(int(a == b) for a, b in zip(data["y_train"], train_preds))
        train_accuracy = (
            (train_correct / len(data["y_train"]) * 100) if data["y_train"] else 0
        )


        raw_test_preds = self.model.predict(data["X_test"])
        test_preds = [(-1 if p == 0 else 1) for p in raw_test_preds]
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
