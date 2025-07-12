import glob
import os
import json
import base64
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, List
import concurrent.futures

from pymongo import MongoClient, errors

# Avoid error
MongoClient.__del__ = lambda self: None

from pymongo.database import Database
from bson import Binary
from PIL import Image as PILImage
from pyrust.src.utils.logger import logger


def get_env_var(name: str, default: str = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        logger.error(f"Environment variable '{name}' not set and no default provided")
        raise ValueError(f"{name} environment variable not set and no default provided")
    return value


class MongoDB:
    def __init__(self, retries: int = 3, delay: float = 2.0):
        self.retries = retries
        self.delay = delay
        self.client = None
        self.db = self._connect_db()

    def _get_uri(self) -> str:
        host = get_env_var("MONGO_HOST", "localhost")
        port = get_env_var("MONGO_PORT", "27017")
        user = os.environ.get("MONGO_USER")
        pwd = os.environ.get("MONGO_PASSWORD")
        if user and pwd:
            uri = f"mongodb://{user}:{pwd}@{host}:{port}/?authSource=admin"
        else:
            uri = f"mongodb://{host}:{port}/"
        logger.debug(f"MongoDB URI constructed: {uri}")
        return uri

    def _connect_db(self) -> Database:
        uri = self._get_uri()
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.info(f"Connecting to MongoDB (attempt {attempt}/{self.retries})")
                self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                self.client.admin.command("ping")
                db_name = get_env_var("MONGO_DATABASE", "admin")
                logger.info(f"Connected to MongoDB database '{db_name}'")
                return self.client[db_name]
            except errors.ServerSelectionTimeoutError as e:
                last_error = e
                logger.warning(f"MongoDB connection attempt {attempt} failed: {e}")
                if attempt < self.retries:
                    sleep_time = self.delay * attempt
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
        logger.error(f"MongoDB connection failed after {self.retries} attempts")
        raise RuntimeError(
            f"MongoDB connection failed after {self.retries} attempts: {last_error}"
        )

    def close(self):
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB client closed")
            except Exception as e:
                logger.warning(f"Error closing MongoDB client: {e}")


class Image:
    def __init__(self, filename: str, data: bytes, metadata: dict):
        self.filename = filename
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_file(
        cls, path: str, label: str, source: str, prompt: Optional[str] = None
    ) -> "Image":
        logger.debug(f"Reading image file '{path}'")
        with open(path, "rb") as f:
            data = f.read()
        with PILImage.open(path) as img:
            width, height = img.size
        metadata = {
            "label": label,
            "source": source,
            "prompt": prompt,
            "format": os.path.splitext(path)[1].lstrip("."),
            "created_at": datetime.now(ZoneInfo("Europe/Paris")),
            "size_bytes": len(data),
            "width": width,
            "height": height,
        }
        logger.info(
            f"Created Image object for '{path}' ({width}x{height}, {len(data)} bytes)"
        )
        return cls(os.path.basename(path), data, metadata)

    @classmethod
    def from_db(
        cls, collection, filename: str, metadata_filter: dict
    ) -> Optional["Image"]:
        logger.debug(
            f"Loading image '{filename}' from DB with filter {metadata_filter}"
        )
        query = {"filename": filename, **metadata_filter}
        doc = collection.find_one(query)
        if not doc:
            logger.warning(
                f"No document found for {filename} with filter {metadata_filter}"
            )
            return None
        logger.info(f"Loaded '{filename}' from DB")
        return cls(doc["filename"], doc["data"], doc["metadata"])

    def save(self, collection) -> str:
        query = {
            "filename": self.filename,
            **{f"metadata.{k}": v for k, v in self.metadata.items()},
        }
        if collection.find_one(query):
            logger.info(f"Image '{self.filename}' already exists in DB, skipping save")
            return self.filename
        doc = {
            "filename": self.filename,
            "data": Binary(self.data),
            "metadata": self.metadata,
        }
        collection.insert_one(doc)
        logger.info(f"Saved image '{self.filename}' to DB")
        return self.filename


class ImageCollection:
    def __init__(self, collection, label: str, max_workers: int = 8):
        self.collection = collection
        self.label = label
        self.max_workers = max_workers

    def load_all(self) -> List[Image]:
        logger.info(f"Listing images with label '{self.label}'")
        entries = list(
            self.collection.find({"metadata.label": self.label}, {"filename": 1})
        )
        logger.info(f"Found {len(entries)} images")
        images: List[Image] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(
                    Image.from_db,
                    self.collection,
                    e["filename"],
                    {"metadata.label": self.label},
                )
                for e in entries
            ]
            for future in concurrent.futures.as_completed(futures):
                img = future.result()
                if img:
                    images.append(img)
        logger.info(f"Loaded {len(images)} images in parallel")
        return images

    def save_folder(self, directory: str, source: str, captions: dict = None) -> None:
        logger.info(f"Saving folder '{directory}' to DB with label '{self.label}'")
        caps = captions or None
        cap_path = os.path.join(directory, "captions.json")
        if caps is None and os.path.exists(cap_path):
            with open(cap_path, "r") as f:
                caps = json.load(f)
        for path in glob.glob(os.path.join(directory, "*")):
            prompt = (
                caps.get(os.path.basename(path), {}).get("caption") if caps else None
            )
            img = Image.from_file(path, self.label, source, prompt)
            try:
                img.save(self.collection)
            except Exception as e:
                logger.error(f"Error saving {path}: {e}")
        logger.info(f"Finished saving folder '{directory}'")

    def export_to_file(self, export_path: str) -> None:
        logger.info(f"Exporting images with label '{self.label}' to '{export_path}'")
        images = self.load_all()
        export_list = []
        for img in images:
            meta = {
                k: (v.isoformat() if isinstance(v, datetime) else v)
                for k, v in img.metadata.items()
            }
            export_list.append(
                {
                    "filename": img.filename,
                    "metadata": meta,
                    "data_base64": base64.b64encode(img.data).decode("utf-8"),
                }
            )
        with open(export_path, "w") as f:
            json.dump(export_list, f)
        logger.info(f"Export completed to {export_path}")

    def import_from_file(self, import_path: str) -> None:
        logger.info(f"Importing images from '{import_path}'")
        with open(import_path, "r") as f:
            import_list = json.load(f)
        for entry in import_list:
            try:
                data = base64.b64decode(entry["data_base64"])
                meta = entry["metadata"]
                if "created_at" in meta:
                    try:
                        meta["created_at"] = datetime.fromisoformat(meta["created_at"])
                    except ValueError:
                        pass
                img = Image(entry["filename"], data, meta)
                img.save(self.collection)
            except Exception as e:
                logger.error(f"Error importing {entry.get('filename')}: {e}")
        logger.info(f"Import completed from {import_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    mongo = MongoDB()
    coll = mongo.db["images"]

    label = os.getenv("IMAGE_LABEL", "real")
    source = os.getenv("IMAGE_SOURCE", "unspecified")
    directory = os.getenv("IMAGE_DIR", "../data/real")

    images = ImageCollection(coll, label)
    # export_file = os.getenv("EXPORT_PATH", "export.json")
    # images.export_to_file(export_file)
    images.save_folder(directory, source)
    # loaded = images.load_all()
    # for img in loaded:
    #     logger.info(
    #         f"{img.filename}: {img.metadata.get('size_bytes')} octets, {img.metadata.get('width')}x{img.metadata.get('height')}"
    #     )
