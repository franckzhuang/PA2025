import csv
import os
from datetime import datetime
from enum import Enum
from PIL import Image
import random

LOG_FILE_NAME = "experiment_log.csv"


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class Logger:
    @staticmethod
    def log_experiment_parameters(
        model: str,
        config: dict,
        len_real_images: int,
        len_ai_images: int,
        total_images: int,
        status: Status,
        error_message: str = "",
        final_accuracy: float = None,
    ):
        write_header = (
            not os.path.isfile(LOG_FILE_NAME) or os.path.getsize(LOG_FILE_NAME) == 0
        )

        headers = [
            "Timestamp",
            "Model_Name",
            "Image_Size_Width",
            "Image_Size_Height",
            "Images_Per_Class_Max_Config",
            "Len_Real_Images_Loaded",
            "Len_AI_Images_Loaded",
            "Total_Images_Used",
            "Epochs",
            "Learning_Rate",
            "Status",
            "Error_Message",
            "Final_Accuracy_Train",
        ]

        data_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model,
            config["image_size"][0],
            config["image_size"][1],
            config["max_images_per_class"],
            len_real_images,
            len_ai_images,
            total_images,
            config["epochs"],
            config["learning_rate"],
            status.value,
            error_message,
            f"{final_accuracy:.2f}%" if final_accuracy is not None else "N/A",
        ]

        try:
            with open(LOG_FILE_NAME, mode="a", newline="") as file:
                writer = csv.writer(file)

                if write_header:
                    writer.writerow(headers)

                writer.writerow(data_row)

            if write_header:
                print(
                    f"New log file '{LOG_FILE_NAME}' created with experiment's parameters."
                )
            else:
                print(f"Experiment's parameters added to '{LOG_FILE_NAME}'")

        except IOError as e:
            print(f"Error while writing log in CSV: {e}")


class ImageUtils:
    @staticmethod
    def preprocess_image(image_path, target_size):
        try:
            img = Image.open(image_path).convert("L")
            img = img.resize(target_size)
            pixel_values = [p / 255.0 for p in list(img.getdata())]
            return pixel_values
        except Exception as e:
            print(f"Error while preprocessing image '{image_path}': {e}")
            return None


class DataLoader:
    @staticmethod
    def load_data(config):
        X_data, y_data, file_names_list = [], [], []
        loaded_counts = {"real": 0, "ai": 0}

        data_sources_config = [
            {
                "path": config["real_images_path"],
                "label": 1.0,
                "key": "real",
            },
            {
                "path": config["ai_images_path"],
                "label": -1.0,
                "key": "ai"
            },
        ]

        print("Loading images...")
        for source in data_sources_config:
            folder_path = source["path"]
            label = source["label"]
            source_key = source["key"]

            if not os.path.exists(folder_path):
                print(f"Dossier {folder_path} non trouvé.")
                continue

            current_loaded_for_class = 0
            image_files = os.listdir(folder_path)
            random.shuffle(image_files)

            for image_name in image_files:
                if current_loaded_for_class >= config["max_images_per_class"]:
                    break
                if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(folder_path, image_name) # Obtenir le chemin complet
                    features = ImageUtils.preprocess_image(
                        full_path, config["image_size"]
                    )
                    if features:
                        X_data.append(features)
                        y_data.append(label)
                        file_names_list.append(image_name)
                        current_loaded_for_class += 1

            loaded_counts[source_key] = current_loaded_for_class
            print(
                f"{current_loaded_for_class} images loaded from {folder_path}."
            )

        if not X_data:
            return [], [], [], loaded_counts["real"], loaded_counts["ai"]

        combined_data = list(zip(X_data, y_data, file_names_list))
        random.shuffle(combined_data)

        if not combined_data:
            return [], [], loaded_counts["real"], loaded_counts["ai"]

        X_data_shuffled, y_data_shuffled, files_shuffled = [list(t) for t in zip(*combined_data)]
        return X_data_shuffled, y_data_shuffled, files_shuffled, loaded_counts["real"], loaded_counts["ai"]
