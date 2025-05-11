import mini_keras as mk

from utils import Logger, Status, DataLoader


# Change this function to implement multiple models
def train_and_evaluate_model(X_train, y_train, file_names, config):
    print("\nTraining LinearModel...")
    model = mk.LinearModel(
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        mode="classification",
        verbose=config["verbose"],
    )

    model.fit(X_train, y_train)
    print("Training finished.")

    predictions = model.predict(X_train)

    correct = sum(1 for i in range(len(y_train)) if y_train[i] == predictions[i])
    accuracy = (correct / len(y_train)) * 100 if len(y_train) > 0 else 0

    print(
        f"\nAccuracy on training dataset : {accuracy:.2f}% ({correct}/{len(y_train)} corrects)"
    )

    if len(X_train) <= config.get("max_predictions_to_show", 10):
        print("-" * 60)
        print(f"{'File Name':<30} | {'Actual':<5} | {'Predicted':<5}")
        print("-" * 60)
        for i in range(len(X_train)):
            actual_str = "Real" if y_train[i] == 1.0 else "AI"
            pred_str = "Real" if predictions[i] == 1.0 else "AI"
            print(f"  {file_names[i]:<30} | {actual_str:<5} | {pred_str:<5}")
    return accuracy


def run_experiment():
    experiment_config = {
        "image_size": (64, 64),
        "max_images_per_class": 90,
        "real_images_path": "data/real",
        "ai_images_path": "data/ai",
        "learning_rate": 0.1,
        "epochs": 100,
        "verbose": False,
        "max_predictions_to_show": 100,
    }

    X_images, y_labels, files_name_loaded, len_real, len_ai = DataLoader.load_data(experiment_config)
    total_images_loaded = len(X_images)

    if total_images_loaded < 2:
        print(
            "Not enough images (less than 2) loaded. Stopping the experiment."
        )
        Logger.log_experiment_parameters(
            model="LinearModel",
            config=experiment_config,
            len_real_images=len_real,
            len_ai_images=len_ai,
            total_images=total_images_loaded,
            status=Status.SUCCESS,
            error_message="Not enough images loaded"
        )
        return

    print(f"\nTotal length images for the experiment: {total_images_loaded}")

    final_accuracy = None
    try:
        final_accuracy = train_and_evaluate_model(X_images, y_labels, files_name_loaded, experiment_config)
    except Exception as e:
        print(f"Error during the experiment : {e}")
    finally:
        Logger.log_experiment_parameters(
            model="LinearModel", # Later, change this with the model name
            config=experiment_config,
            len_real_images=len_real,
            len_ai_images=len_ai,
            total_images=total_images_loaded,
            status=Status.SUCCESS,
            final_accuracy=final_accuracy,
        )


if __name__ == "__main__":
    run_experiment()
