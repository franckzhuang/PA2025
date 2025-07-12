import os
from PIL import Image

def resize_images(input_dir, output_dir, max_size=984):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")):
            input_path = os.path.join(input_dir, filename)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, base_name + ".jpg")

            with Image.open(input_path) as img:
                w, h = img.size
                max_dim = max(h, w)

                if max_dim > max_size:
                    ratio = max_size / max_dim
                    new_w = int(w * ratio)
                    new_h = int(h * ratio)
                    img = img.resize((new_w, new_h), Image.LANCZOS)

                if img.mode in ("RGBA", "LA"):
                    img = img.convert("RGB")

                img.save(output_path, format="JPEG")
                print(f"Image saved: {output_path}")


if __name__ == "__main__":
    input_folder = "pyrust/src/data/real_no_resized"
    output_folder = "pyrust/src/data/real"
    resize_images(input_folder, output_folder)