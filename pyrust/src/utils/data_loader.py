import os
import random
import base64
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from io import BytesIO

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from pyrust.src.database.mongo import MongoDB
from pyrust.src.utils.env import get_env_var
from pyrust.src.utils.logger import logger

MAX_WORKERS = int(get_env_var("MAX_WORKERS", None))

class LoaderType(Enum):
    LOCAL = "local"
    MONGO = "mongo"


class ImageUtils:
    @staticmethod
    def preprocess_image(image_path, target_size):
        try:
            img = Image.open(image_path).convert("RGB").resize(target_size)
            # Conversion en tableau numpy et normalisation
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Aplatir le tableau et le convertir en liste
            return img_array.flatten().tolist()
        except Exception as e:
            logger.error(f"Error processing '{image_path}': {e}")
            return None

    @staticmethod
    def preprocess_image_wrapper(args):
        path, target_size = args
        return ImageUtils.preprocess_image(path, target_size)

    @staticmethod
    def preprocess_bytes(image_bytes, target_size):
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB").resize(target_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array.flatten().tolist()
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            return None


class DataLoader:
    @staticmethod
    def load_data(config):
        loader = get_env_var("DATA_LOADER", LoaderType.LOCAL.value).lower()
        logger.info(f"Loading data using loader: %s", loader)
        if loader == LoaderType.LOCAL.value:
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

        tasks = []
        task_info = {}

        logger.info("Preparing image processing tasks...")
        for source in sources:
            folder, label, key = source["path"], source["label"], source["key"]
            if not os.path.exists(folder):
                logger.warning(f"Folder '{folder}' not found.")
                continue

            all_files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            files_to_process = random.sample(all_files, min(len(all_files), max_per_class))

            for fname in files_to_process:
                path = os.path.join(folder, fname)
                tasks.append((path, target_size))
                task_info[path] = {"label": label, "fname": fname, "key": key}

        logger.info(f"Processing {len(tasks)} images in parallel...")
        with ProcessPoolExecutor() as executor:
            results = executor.map(ImageUtils.preprocess_image_wrapper, tasks)

        for task_args, features in zip(tasks, results):
            if features:
                path = task_args[0]
                info = task_info[path]
                X_data.append(features)
                y_data.append(info["label"])
                file_names_list.append(info["fname"])
                loaded_counts[info["key"]] += 1

        logger.info(f"Loaded counts: {loaded_counts}")
        return DataLoader._prepare_split(X_data, y_data, file_names_list, loaded_counts)

    @staticmethod
    def _load_mongo(config):
        X_data, y_data, file_names_list = [], [], []
        loaded_counts = {"real": 0, "ai": 0}
        max_per_class = config.get("max_images_per_class", 100)
        target_size = tuple(config.get("image_size", ()))
        mongodb = MongoDB()
        coll = mongodb.db["images"]
        sources = [
            {"label_val": "real", "label": 1.0, "key": "real"},
            {"label_val": "ai", "label": -1.0, "key": "ai"},
        ]
        logger.info("Sources for MongoDB: %s", sources)
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
