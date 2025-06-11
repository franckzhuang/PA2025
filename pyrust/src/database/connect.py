import glob
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from pymongo import MongoClient, errors
from pymongo.database import Database
from gridfs import GridFS
from dotenv import load_dotenv

from pyrust.src.utils import EnvUtils

load_dotenv()


class MongoDB:
    def __init__(self):
        self.db = self.connect_db()
        self.fs = self.connect_gridfs()

    def get_mongo_uri(self) -> str:
        host = EnvUtils.get_env_var("MONGO_HOST", "localhost")
        port = EnvUtils.get_env_var("MONGO_PORT", "27017")
        user = os.environ.get("MONGO_USER")
        password = os.environ.get("MONGO_PASSWORD")

        if user and password:
            return f"mongodb://{user}:{password}@{host}:{port}/?authSource=admin"
        else:
            return f"mongodb://{host}:{port}/"


    def connect_db(self) -> Database:
        uri = self.get_mongo_uri()
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            print("Connected to MongoDB")
        except errors.ServerSelectionTimeoutError as e:
            raise RuntimeError(f"Could not connect to MongoDB: {e}")

        database_name = EnvUtils.get_env_var("MONGO_DATABASE", "admin")
        return client[database_name]


    def connect_gridfs(self) -> GridFS:
        return GridFS(self.db)

    def insert_image_from_file(self, filepath, label, source, prompt=None):
        with open(filepath, "rb") as f:
            data = f.read()

        filename = os.path.basename(filepath)

        if self.fs.exists({"filename": filename, "label": label}):
            print(f"'{filename}' with label '{label}' already exists, skipping insertion.")
            return

        try:
            self.fs.put(data, filename=filename, metadata={
                "label": label,
                "source": source,
                "prompt": prompt,
                "format": "jpeg",
                "created_at": datetime.now(ZoneInfo("Europe/Paris"))
            })
        except Exception as e:
            print(f"Error inserting {filename}: {e}")
            return

    def insert_batch_from_dir(self, image_dir: str, label: str, source: str, prompt=None):
        captions = None
        if label == "ai":
            with open(f"{image_dir}/captions.json", "r") as f:
                captions = json.load(f)
        for path in glob.glob(f"{image_dir}/*.png"):
            try:
                self.insert_image_from_file(
                    filepath=path,
                    label=label,
                    source=source,
                    prompt=captions[os.path.basename(path)]["caption"] if captions else None
                )
            except:
                print(f"Error processing {path}, skipping.")
                continue
        print("Batch insertion completed.")

    def get_image(self, filename: str, label: str):
        file = self.fs.find_one({"filename": filename, "metadata.label": label})
        if not file:
            print(f"No file found with filename '{filename}' and label '{label}'")
            return None

        img_data = file.read()
        return img_data

    def get_batch_images(self, label: str):
        files = self.fs.find({"metadata.label": label})
        images = []
        for file in files:
            img_data = file.read()
            images.append({
                "filename": file.filename,
                "data": img_data,
                "metadata": file.metadata
            })
        return images

    # def read(self, filename: str, label: str):
    #     file = self.fs.find_one({"filename": filename, "metadata.label": label})
    #     img_data = file.read()
    #
    #     img = Image.open(io.BytesIO(img_data))
    #     img.show()
