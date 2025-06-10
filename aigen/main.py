import os
import torch
import json
from utils import load_blip2_model, load_sd_pipeline, generate_caption, generate_image_from_speech
from PIL import Image

def main():
    input_folder = input(
        "Enter the path to the folder containing the images: ").strip()

    if not os.path.isdir(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return

    output_folder = os.path.join(input_folder, "generated_images")
    os.makedirs(output_folder, exist_ok=True)

    captions_file = os.path.join(input_folder, "captions.json")

    # Check if captions.json already exists
    if os.path.exists(captions_file):
        user_choice = input(
            "A 'captions.json' file already exists. Do you want to skip caption generation? (y/n): ").strip().lower()
        if user_choice == 'y':
            print("Skipping caption generation phase.")
            with open(captions_file, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
        else:
            captions_dict = generate_captions(input_folder, captions_file)
    else:
        captions_dict = generate_captions(input_folder, captions_file)

    # Release BLIP-2 to free memory
    torch.cuda.empty_cache()

    # Load Stable Diffusion and generate images
    print("\nStarting image generation phase...\n")
    sd_pipe = load_sd_pipeline()

    for idx, (img_name, img_stat) in enumerate(captions_dict.items(), 1):
        output_image_path = os.path.join(
            output_folder, f"{os.path.splitext(img_name)[0]}.jpg")
        print(f"[{idx}/{len(captions_dict)}] Generating image for: {img_name}")

        try:
            generate_image_from_speech(sd_pipe, img_stat['caption'], img_stat['height'], img_stat['width'], output_image_path)
        except Exception as e:
            print(f"Error generating image for {img_name}: {e}")

    print("\nProcessing complete.")


def generate_captions(input_folder, captions_file):
    print("\nStarting caption generation phase...\n")
    processor, blip2_model = load_blip2_model()

    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    images = [f for f in os.listdir(
        input_folder) if f.lower().endswith(supported_formats)]

    if not images:
        print("Error: No valid images found in the folder.")
        exit()

    captions_dict = {}

    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(input_folder, img_name)
        print(f"[{idx}/{len(images)}] Processing image: {img_name}")

        try:
            # Get image dimensions
            with Image.open(img_path) as img:
                height, width = img.size
            caption = generate_caption(img_path, processor, blip2_model)
            captions_dict[img_name] = {'caption' : caption, 'height' : width, 'width' : height}
            print(f"Caption: {caption}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    with open(captions_file, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=4, ensure_ascii=False)

    print(f"\nCaptions saved to {captions_file}")

    # Release BLIP-2 after captioning
    del blip2_model
    del processor
    torch.cuda.empty_cache()

    return captions_dict


if __name__ == "__main__":
    main()
