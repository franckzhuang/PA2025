import os
import random
from enum import Enum
from PIL import Image
from sklearn.model_selection import train_test_split


class ImageUtils:
    @staticmethod
    def preprocess_image(image_path, target_size):
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(target_size)
            pixel_values = list(img.getdata())
            return [c / 255.0 for pixel in pixel_values for c in pixel]
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
                    full_path = os.path.join(folder_path, image_name)
                    features = ImageUtils.preprocess_image(
                        full_path, config["image_size"]
                    )
                    if features:
                        X_data.append(features)
                        y_data.append(label)
                        file_names_list.append(image_name)
                        current_loaded_for_class += 1
            loaded_counts[source_key] = current_loaded_for_class
            print(f"{current_loaded_for_class} images loaded from {folder_path}.")
        if not X_data:
            return {
                "X_train": [],
                "X_test": [],
                "y_train": [],
                "y_test": [],
                "files_train": [],
                "files_test": [],
                "loaded_counts": loaded_counts,
            }
        combined_data = list(zip(X_data, y_data, file_names_list))
        random.shuffle(combined_data)
        if not combined_data:
            return {
                "X_train": [],
                "X_test": [],
                "y_train": [],
                "y_test": [],
                "files_train": [],
                "files_test": [],
                "loaded_counts": loaded_counts,
            }
        X_data_shuffled, y_data_shuffled, files_shuffled = [
            list(t) for t in zip(*combined_data)
        ]
        X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
            X_data_shuffled,
            y_data_shuffled,
            files_shuffled,
            test_size=0.2,
            random_state=42,
            stratify=y_data_shuffled,
        )
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "files_train": files_train,
            "files_test": files_test,
            "loaded_counts": loaded_counts,
        }


class EnvUtils:
    @staticmethod
    def get_env_variable(var_name, default_value=None):
        return os.getenv(var_name, default_value)

    @staticmethod
    def set_env_variable(var_name, value):
        os.environ[var_name] = value
        print(f"Environment variable '{var_name}' set to '{value}'.")

    def get_env_var(name: str, default: str = None) -> str:
        value = os.environ.get(name, default)
        if value is None:
            raise ValueError(
                f"{name} environment variable not set and no default provided"
            )
        return value


# LOG_FILE_NAME = "experiment_log.csv"

# class Logger:
#     @staticmethod
#     def log_experiment_parameters(
#         model: str,
#         config: dict,
#         len_real_images: int,
#         len_ai_images: int,
#         total_images: int,
#         status: Status,
#         error_message: str = "",
#         final_accuracy: float = None,
#     ):
#         write_header = (
#             not os.path.isfile(LOG_FILE_NAME) or os.path.getsize(LOG_FILE_NAME) == 0
#         )
#
#         headers = [
#             "Timestamp",
#             "Model_Name",
#             "Image_Size_Width",
#             "Image_Size_Height",
#             "Images_Per_Class_Max_Config",
#             "Len_Real_Images_Loaded",
#             "Len_AI_Images_Loaded",
#             "Total_Images_Used",
#             "Epochs",
#             "Learning_Rate",
#             "Status",
#             "Error_Message",
#             "Final_Accuracy_Train",
#         ]
#
#         data_row = [
#             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             model,
#             config["image_size"][0],
#             config["image_size"][1],
#             config["max_images_per_class"],
#             len_real_images,
#             len_ai_images,
#             total_images,
#             config.get("epochs", "N/A"),
#             config.get("learning_rate", "N/A"),
#             status.value,
#             error_message,
#             f"{final_accuracy:.2f}%" if final_accuracy is not None else "N/A",
#         ]
#
#         try:
#             with open(LOG_FILE_NAME, mode="a", newline="") as file:
#                 writer = csv.writer(file)
#
#                 if write_header:
#                     writer.writerow(headers)
#
#                 writer.writerow(data_row)
#
#             if write_header:
#                 print(
#                     f"New log file '{LOG_FILE_NAME}' created with experiment's parameters."
#                 )
#             else:
#                 print(f"Experiment's parameters added to '{LOG_FILE_NAME}'")
#
#         except IOError as e:
#             print(f"Error while writing log in CSV: {e}")
