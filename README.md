# 📍 AGID - Anti Generated Image Detection

AGID is a full-stack application designed to train, evaluate, and manage machine learning models that detect whether an image is real or AI-generated. Built with Python, Streamlit, and FastAPI, the app provides both interactive training and powerful detection capabilities using custom models (SVM, MLP, Linear Classifier, KMeans).


## 🔮 Stack

![Static Badge](https://img.shields.io/badge/python-gray?style=for-the-badge&logo=Python)
![Static Badge](https://img.shields.io/badge/pandas-gray?style=for-the-badge&logo=Pandas)
![Static Badge](https://img.shields.io/badge/streamlit-gray?style=for-the-badge&logo=Streamlit)
![Static Badge](https://img.shields.io/badge/rust-gray?style=for-the-badge&logo=Rust)

## 🌐 Project Overview

### 📁 Project Structure

The key files and directories related to the frontend are structured as follows:

```txt
PA2025/
├── aigen/              
│   ├── main.py   
│   ├── utils.py          
│   └── valid_data.py        
├── front/           
│   ├── src/           
│   │   ├── pages/
│   │   ├── 0_Detector.py  
│   │   ├── ui.py
│   │   └── utils.py
│   └── Dockerfile.py          
├── pyrust/
│   ├── src/
│   │   ├── api/
│   │   │   ├── controllers/
│   │   │   ├── service/
│   │   │   ├── app.py
│   │   │   ├── models.py
│   │   │   ├── schemas.py
│   │   ├── data/
│   │   ├── database/
│   │   ├── scripts/
│   │   ├── training_models/
│   │   ├── utils/
├── mini-keras/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── kmeans.rs
│   │   ├── linear_models.rs
│   │   ├── mlp.rs
│   │   ├── rbf.rs
│   │   ├── rbf_naive.rs
│   │   ├── svm.rs
│   │   ├── utils.rs
│   └── Cargo.toml
├── scraper/
│   ├── src/
│   │   ├── main.py
│   │   ├── utils.py
│   │   ├── models.py
│   └── Dockerfile
├── requirements.txt
├── .env
├── README.md
└── .gitignore

```

## 🚀 Features

#### 🧱 Dataset Construction Stages

###### Non-AI Generated Images:
- Bots & Web scrapers : Use web scrapers and bots to gather non-AI generated images on the internet
- Taking non-AI generated pictures manually (Phone, cameras, screenshots?, ...)
- Algorithmic validation of images : check valid formats, extensions (ex. no webp, no svg, etc...) and other details
- Human validation : Human review of each image to validate usefulness and or eliminate if could bias the models

###### AI Generated images:
- Use **multiple** open source image generation models as our dataset (multiple to reduce biases). Run models overnight
- Manually search for AI-generated images from trusted sources (Midjourney, ...)
- Human validation : Human review of each image to validate usefulness and or eliminate if could bias the models

#### 🧪 Detection
- Upload an image and predict whether it's AI-generated or real.
- Uses trained models with different ML techniques (SVM, MLP, etc.).

#### 🏋️‍♀️ Model Training
- Start training jobs directly from the Streamlit interface.
- Customize hyperparameters (e.g., learning rate, kernel, hidden layers).
- Visual feedback: accuracy, pie charts, training progress bar, metrics.

#### 📖 History & Management
- View all completed jobs and saved models.
- Download model parameters.
- Evaluate models on new data.
- Import/export models to/from JSON.

## 📡 API Endpoints

| Method | Endpoint                     | Description                                      |
|--------|------------------------------|--------------------------------------------------|
| GET    | /evaluate/models             | List all saved models, grouped by type           |
| POST   | /evaluate/run                | Evaluate an existing model with input data       |
| GET    | /evaluate/saved_models       | Retrieve all saved models (name, job_id, id)     |
| POST   | /evaluate/save_model         | Save a model from a job_id                       |
| GET    | /models/details/{model_name} | Detail a saved model with job info               |
| POST   | /train/linear_classification | Start training a linear classification model     |
| POST   | /train/mlp                   | Start training an MLP model                      |
| POST   | /train/svm                   | Start training an SVM model                      |
| POST   | /train/rbf                   | Start training an RBF model                      |
| GET    | /train/status/{job_id}       | Get the status of a training job                 |
| GET    | /train/{job_id}/params       | Get hyperparameters of a job from a file         |
| GET    | /train                       | List the last 100 training jobs (history)        |
| POST   | /training/import_model       | Manually import a model with parameters and metadata

## ⚙️ Getting Started

System Requirements:

- Python 3.10+
- Docker
- Package managers : pip, npm
- MongoDB 

## 🛰️ Environnement Variables

These environnement variables are required for backend and frontend to work properly. Create a `.env` file in the root directory of your project and add the following variables:

```env
LOG_LEVEL=INFO

# SCRAPER
SEARCH=
SCROLL_TIMEOUT=30
LUMMMI_SCRAPER=false
PINTEREST_SCRAPER=false
UNSPLASH_SCRAPER=false
PEXELS_SCRAPER=false
PIXABAY_SCRAPER=false
STOCKSNAP_SCRAPER=false

# Options: local, mongo
DATA_LOADER=local

# MONGO
MONGO_HOST=
MONGO_PORT=
MONGO_DATABASE=
MONGO_USER=
MONGO_PASSWORD=

API_URL=
```

## 💻 Installation

### Build Data via Scrapping and AI Gen

#### 🔍 Scraper

> Build the scraper to collect non-AI generated images

```bash
$ cd scraper
$ docker build -t scraper .
$ docker run -d --name scraper scraper
```

This will start the scraper in a Docker container. You can modify the scraper settings in `scraper/src/main.py` to adjust the scraping behavior.

#### 🤖 AI Gen
> Generate AI images from the scrapped folder : In this project, we use a 1 to 1 ratio of AI generated images to non-AI generated images with similar characteristics.

```bash
$ cd aigen
$ python main.py
```

Follow the steps in `aigen/main.py` to generate AI images. You can customize the generation parameters as needed.

### 📦 From source

> Clone the repository

```bash
$ git clone https://github.com/franckzhuang/PA2025
$ cd PA2025
```

> Install Python Dependencies

```bash
$ python -m venv env
$ env/bin/activate  # On Windows, use `env\Scripts\activate`
$ pip install --no-deps -r requirements.txt

$ cd mini-keras
$ pip install maturin
$ maturin build
# Then install the wheel file generated in the `target/wheels` directory
```

> Launch API from project root directory:

```bash
uv run uvicorn pyrust.src.api.app:app --host 0.0.0.0 --port 8000 --reload

# or on pip : uvicorn pyrust.src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

> Run Docker Container for Frontend and DB.

```bash
$ docker compose up --build
```

> You can now access the frontend at `http://localhost:5001` and the API at `http://localhost:8000`.

## 🧑‍💻 How it works

When you run the app, you will see in the sub menu the following options:
- **Detector**: Upload an image to check if it's AI-generated or real.
- **Train**: Start training a new model with custom hyperparameters.
- **History**: View all training jobs and saved models.

#### 🧪 Detector
- Upload an image and click "Predict" to see if it's AI-generated.
- You need at least one model trained and saved to use this feature.
- The app will display the prediction result and the model used for inference.

#### 🏋️‍♂️ Train
- Select a model type (SVM, MLP, etc.) and set hyperparameters.
- Click "Start Training" to begin the training job.
- The app will show training progress, accuracy, and other metrics.
- Once training is complete, you can export the model for later use or transfer it to another user with this app.

#### 📖 History
- View all completed training jobs and saved models.
- Download model parameters in JSON format.
- Save models for uses in Detection page.


## 👨‍🎓 Author

- [@Huang-Frederic](https://github.com/Huang-Frederic)
- [@Zhuang-Franck](https://github.com/franckzhuang)
- [@Charara-Mohamad](https://github.com/charara-code)


<!-- ## 🔗 Acknowledgements

Usefull Documentations :

-->
