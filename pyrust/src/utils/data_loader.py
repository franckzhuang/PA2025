import os
import random
import json
import base64
from enum import Enum
from io import BytesIO

from PIL import Image
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

from pyrust.src.database.mongo import get_env_var, MongoDB
from pyrust.src.utils.logger import logger


class LoaderType(Enum):
    LOCAL = "local"
    MONGO = "mongo"


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

    @staticmethod
    def preprocess_bytes(image_bytes, target_size):
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img = img.resize(target_size)
            pixel_values = list(img.getdata())
            return [c / 255.0 for pixel in pixel_values for c in pixel]
        except Exception as e:
            print(f"Error while preprocessing image bytes: {e}")
            return None


class DataLoader:
    @staticmethod
    def load_data(config):
        loader = get_env_var("DATA_LOADER", LoaderType.LOCAL.value).lower()
        logger.info(f"Loading data using loader: %s", loader)
        if loader == LoaderType.LOCAL:
            return DataLoader._load_local(config)
        else:
            return DataLoader._load_mongo(config)

    @staticmethod
    def _load_local(config):
        X_data, y_data, file_names_list = [], [], []
        loaded_counts = {"real": 0, "ai": 0}
        max_per_class = config.get("max_images_per_class", 0)
        target_size = tuple(config.get("image_size", ()))
        sources = [
            {"path": config["real_images_path"], "label": 1.0, "key": "real"},
            {"path": config["ai_images_path"], "label": -1.0, "key": "ai"},
        ]
        print("Loading images from local folders...")
        for source in sources:
            folder, label, key = source["path"], source["label"], source["key"]
            if not os.path.exists(folder):
                print(f"Folder '{folder}' not found.")
                continue
            files = os.listdir(folder)
            random.shuffle(files)
            count = 0
            for fname in files:
                if count >= max_per_class:
                    break
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(folder, fname)
                    features = ImageUtils.preprocess_image(path, target_size)
                    if features:
                        X_data.append(features)
                        y_data.append(label)
                        file_names_list.append(fname)
                        count += 1
            loaded_counts[key] = count
            logger.info(f"{count} images loaded from '{folder}' (label={label}).")
        return DataLoader._prepare_split(X_data, y_data, file_names_list, loaded_counts)

    @staticmethod
    def _load_mongo(config):
        X_data, y_data, file_names_list = [], [], []
        loaded_counts = {"real": 0, "ai": 0}
        max_per_class = config.get("max_images_per_class", 0)
        target_size = tuple(config.get("image_size", ()))
        mongodb = MongoDB()
        coll = mongodb.db["images"]
        sources = [
            {"label_val": "real", "label": 1.0, "key": "real"},
            {"label_val": "ai", "label": -1.0, "key": "ai"},
        ]
        for source in sources:
            label_val, label, key = source["label_val"], source["label"], source["key"]
            cursor = coll.find({"metadata.label": label_val}).limit(max_per_class)
            count = 0
            for doc in cursor:
                data_bytes = doc.get("data") or doc.get("data_base64")
                if isinstance(data_bytes, str):
                    data_bytes = base64.b64decode(data_bytes)
                features = ImageUtils.preprocess_bytes(data_bytes, target_size)
                if features:
                    X_data.append(features)
                    y_data.append(label)
                    file_names_list.append(doc.get("filename"))
                    count += 1
            loaded_counts[key] = count
            logger.info(
                f"{count} images loaded from Mongo collection 'images' for label '{label_val}'."
            )
        return DataLoader._prepare_split(X_data, y_data, file_names_list, loaded_counts)

    @staticmethod
    def _prepare_split(X_data, y_data, file_names_list, loaded_counts):
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
        combined = list(zip(X_data, y_data, file_names_list))
        random.shuffle(combined)
        X_shuf, y_shuf, files_shuf = zip(*combined)
        X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
            list(X_shuf),
            list(y_shuf),
            list(files_shuf),
            test_size=0.2,
            random_state=42,
            stratify=list(y_shuf),
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
