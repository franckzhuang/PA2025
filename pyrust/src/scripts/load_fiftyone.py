import time

import fiftyone as fo
import os

root_dir = "data/"

dataset = fo.Dataset(name="Landscape Dataset", overwrite=True)

count = 0

for label in ["ai", "real"]:
    folder = os.path.join(root_dir, label)
    if not os.path.isdir(folder):
        print(f"The folder {folder} doesn't exist, it will be ignored.")
        continue
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            file_path = os.path.join(folder, filename)
            sample = fo.Sample(filepath=file_path, label=label)
            dataset.add_sample(sample)
            count += 1

print(f"{count} images added to dataset.")

session = fo.launch_app(dataset)

try:
    print("Ctrl+C to leave (or stop the container)")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Closing the FiftyOne session...")
