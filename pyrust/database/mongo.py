from zoneinfo import ZoneInfo

from gridfs import GridFS
from datetime import datetime, timezone
import os
import glob
from PIL import Image
import io

from pyrust.database.connect import connect_gridfs


def insert_image(fs: GridFS, filepath, label, source, prompt=None):
    with open(filepath, "rb") as f:
        data = f.read()

    filename = os.path.basename(filepath)

    if fs.exists({"filename": filename}):
        print(f"'{filename}' already exists, skipping insertion.")
        return

    try:
        fs.put(data, filename=filename, metadata={
            "label": label,
            "source": source,
            "prompt": prompt,
            "format": "jpeg",
            "created_at": datetime.now(ZoneInfo("Europe/Paris"))
        })
    except Exception as e:
        print(f"Error inserting {filename}: {e}")
        return

    # print(f"Inserted {filename} with ID: {file_id}")

def read(fs: GridFS, filename: str, label: str):
    file = fs.find_one({"filename": filename, "metadata.label": label})
    img_data = file.read()

    img = Image.open(io.BytesIO(img_data))
    img.show()


if __name__ == "__main__":
    fs = connect_gridfs()
    # image_dir = "../data/ai"
    # print(glob.glob(f"{image_dir}/*.png"))
    # for path in glob.glob(f"{image_dir}/*.png"):
    #     print(f"Processing {path}")
    #     insert_image(
    #         fs=fs,
    #         filepath=path,
    #         label="ai" if "ai" in path else "real",
    #         source="stable-diffusion" if "ai_" in path else "flickr",
    #         prompt="example prompt"
    #     )
    read(fs, "UNSPLASH-IMG_2.png", "ai")
