# Hate Speech Inception Detection Project

This project implements a hate speech detection system using a pre-trained language model (RoBERTa) for classification. The workflow includes data processing, model training, and serving the model via an API for predictions.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Running the API](#running-the-api)
- [Making Predictions](#making-predictions)
  - [Using `predict_model.py`](#using-predict_modelpy)
  - [Using `curl`](#using-curl)
- [Additional Information](#additional-information)

## Project Structure

```
├── app.py                # API server for model inference
├── data/                 # Data folder (contains raw & processed data)
├── process_data.py       # Data processing script
├── train_model.py        # Model training script
├── predict_model.py      # Script to send post request to the API for prediction
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/utkarsh-iitbhu/Inception-analyser.git
```

### 2. Install the Required Packages
Set up a virtual environment and install the dependencies using the `requirements.txt` file:
```bash
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Processing

You will need to process the data first. The `process_data.py` script processes the dataset, combines tweet files with their annotations, and creates the required CSV files for training and testing.

To run data processing:

```bash
python process_data.py
```

This will:
- Read and combine the raw tweet text and their annotations.
- Save the processed data as `processed_train.csv` and `processed_test.csv` inside the `data/` folder.

## Model Training

Once the data is processed, you can train the model. The `train_model.py` script will train the hate speech detection model and save the best version based on the evaluation metrics.

To run model training:

```bash
python train_model.py
```

This will:
- Train the RoBERTa model on the processed training data.
- Evaluate it on the test data.
- Save the trained model and tokenizer in the `models/` folder.

## Running the API

After the model has been trained, you can serve it using the API defined in `app.py`.

To run the API:

```bash
python app.py
```

This will start a Flask server (by default on `http://127.0.0.1:8080`) and expose an endpoint `/predict` for making predictions.

## Making Predictions

You can now make predictions using the model either by running the `predict_model.py` script or using `curl` commands to hit the `/predict` route on the API.

### Using `predict_model.py`

The `predict_model.py` script sends a POST request to the running API server and retrieves the prediction.

To make a prediction:

```bash
python predict_model.py
```

This will:
- Send a sample request to the running server.
- Print the response, which includes the predicted label (`hate` or `noHate`).

### Using `curl`

You can also send a POST request using `curl` directly to the `/predict` route.

Example:

```bash
curl -X POST http://127.0.0.1:8080/predict \
-H "Content-Type: application/json" \
-d '{"text": "Your input text for hate speech detection here"}'
```

This will return the prediction for the provided input text.

## Additional Information

- Make sure the `data` folder exists even if it's empty; otherwise, the folder won't be tracked in Git.
- The API server runs by default on `http://127.0.0.1:8080`. You can modify the host and port inside `app.py` if needed.
- `predict_model.py` uses Python's `requests` library to send POST requests.
