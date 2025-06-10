import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import StableDiffusionPipeline
from PIL import Image

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}")

# Constants
BLIP2_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
BLIP2_DIR = "./blip2_flant5xl_local"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Load BLIP-2 model with auto-download if missing


def load_blip2_model():
    if not os.path.exists(BLIP2_DIR):
        print("BLIP-2 model not found locally. Downloading from Hugging Face...")
        processor = Blip2Processor.from_pretrained(BLIP2_MODEL_NAME)
        processor.save_pretrained(BLIP2_DIR)

        model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL_NAME)
        model.save_pretrained(BLIP2_DIR)

        print("BLIP-2 model downloaded and saved locally.")
    else:
        print("BLIP-2 model found locally.")

    processor = Blip2Processor.from_pretrained(BLIP2_DIR)
    model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_DIR).to(device)
    return processor, model

# Load Stable Diffusion pipeline


def load_sd_pipeline():
    try:
        print("Loading Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            revision=None,
        ).to(device)
        return pipe
    except Exception as e:
        raise RuntimeError(f"Error loading Stable Diffusion: {e}")

# Generate caption from an image


def generate_caption(image_path, processor, model):
    image = Image.open(image_path).convert('RGB')

    inputs = processor(images=image, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        do_sample=True,
        top_p=1.0,
        temperature=1.0,
        max_new_tokens=100,
    )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Generate an image from a speech (caption)


def generate_image_from_speech(pipe, speech_text,img_height,img_width, output_path):
    print(f"Generating image for: '{speech_text}'")

    # Ensure height and width are divisible by 8
    def round_to_multiple_of_8(x):
        return (x // 8) * 8
    
    sd_height = round_to_multiple_of_8(img_height)
    sd_width = round_to_multiple_of_8(img_width)

    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        image = pipe(prompt=speech_text,height=sd_height, width=sd_width, guidance_scale=8.5).images[0]

    image.save(output_path)
    print(f"Image saved to: {output_path}")
