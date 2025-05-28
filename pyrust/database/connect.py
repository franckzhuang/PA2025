import os
from pymongo import MongoClient, errors
from pymongo.database import Database
from gridfs import GridFS
from dotenv import load_dotenv

load_dotenv()

def get_env_var(name: str, default: str = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ValueError(f"{name} environment variable not set and no default provided")
    return value


def get_mongo_uri() -> str:
    host = get_env_var("MONGO_HOST", "localhost")
    port = get_env_var("MONGO_PORT", "27017")
    user = os.environ.get("MONGO_USER")
    password = os.environ.get("MONGO_PASSWORD")

    if user and password:
        return f"mongodb://{user}:{password}@{host}:{port}/?authSource=admin"
    else:
        return f"mongodb://{host}:{port}/"


def connect_db() -> Database:
    uri = get_mongo_uri()
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print("Connected to MongoDB")
    except errors.ServerSelectionTimeoutError as e:
        raise RuntimeError(f"Could not connect to MongoDB: {e}")

    database_name = get_env_var("MONGO_DATABASE", "admin")
    return client[database_name]


def connect_gridfs() -> GridFS:
    db = connect_db()
    return GridFS(db)
