# 📍 Image-to-Image Captioning and Regeneration Pipeline

This project provides a fully automated pipeline to generate text descriptions (captions) for a batch of images and then regenerate new images from these captions using AI models.

The pipeline is designed for **speed, scalability, and local execution**, capable of handling **thousands of images** efficiently.

## 🔮 Stack

![Static Badge](https://img.shields.io/badge/Python-gray?style=for-the-badge&logo=Python)
![Static Badge](https://img.shields.io/badge/Torch-gray?style=for-the-badge&logo=pytorch)
![Static Badge](https://img.shields.io/badge/Transformers-gray?style=for-the-badge&logo=HuggingFace)
![Static Badge](https://img.shields.io/badge/Diffusers-gray?style=for-the-badge&logo=HuggingFace)
![Static Badge](https://img.shields.io/badge/TQDM-gray?style=for-the-badge&logo=Progress)
![Static Badge](https://img.shields.io/badge/PIL-gray?style=for-the-badge&logo=Pillow)

## 🚀 Getting Started

### System Requirements:

- Python 3.10+
- GPU (recommended) or CPU
- Package manager: pip

## 💻 Installation

### From source

> Clone the repository

```bash
git clone https://github.com/franckzhuang/PA2025.git
cd PA2025/aigen
```

> Install Python Dependencies

```bash
$ python -m venv env
# Activate the environment
# On Linux/Mac
$ source env/bin/activate
# On Windows
$ env\Scripts\activate
# Then install from aigen requirements
$ pip install -r requirements.txt
```

> Run the project

```bash
$ python main.py
```

## 🛐️ How it works

1. The script will try to detect if there is a cuda GPU available to use, then set the device.
1. It will then asks for the path to the folder containing your images.
1. It generates a new caption for each image using BLIP-2, then save it in a `captions.json` file .
   > [!NOTE]  
   > If there is already a `captions.json` in the selected folder, it will ask to skip the caption generation.
1. After generating all captions, BLIP-2 is unloaded from memory to free up GPU/CPU resources.
1. The script then loads Stable Diffusion and regenerates a new image based on each caption.
1. New images are saved inside a `generated_images/` folder inside your selected directory, with the same name file as it's input.

## 📺 Project Structure

| File               | Purpose                                                             |
| :----------------- | :------------------------------------------------------------------ |
| `main.py`          | Main script orchestrating the full process                          |
| `utils.py`         | Helper functions to load models and handle image/caption generation |
| `requirements.txt` | All the requirements needed for the scri                            |
| `README.md`        | That's what you are reading                                         |

## ⚡ Features

- Full local execution excluding download of the models (no external APIs required)
- Automatic model downloading if missing
- Memory optimization by unloading unused models
- TQDM progress bars for visual tracking
- Resume capability via saved `captions.json`
