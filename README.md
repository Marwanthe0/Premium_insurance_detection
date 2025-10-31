# Insurance Premium Category Detection

A concise project that detects insurance premium categories using a FastAPI backend, a Streamlit frontend, Pandas for data processing, and a scikit-learn RandomForest model.

## Features
- Predict insurance premium category from user-provided features
- Batch and single-record prediction endpoints
- Reproducible training pipeline and evaluation metrics
- Simple Streamlit UI for demo and manual testing

## Prerequisites
- Python 3.8+
- pip

## Installation
```sh
pip install -r requirements.txt
```

### Dependencies
- fastapi
- uvicorn
- streamlit
- pandas
- scikit-learn
- joblib

## Running the Application

### Backend (FastAPI)
```sh
uvicorn main:app --reload
```
- API available at http://127.0.0.1:8000
- API documentation at http://127.0.0.1:8000/docs

### Frontend (Streamlit)
```sh
streamlit run app.py
```
- Opens browser with interactive UI for predictions

## Model Details
- RandomForest classifier saved as `model.pkl`
- Model loading and predictions handled by FastAPI backend
- Update predictions by replacing `model.pkl` after retraining

## Data Processing & Training
- Data cleaning and preparation using pandas
- Model training with scikit-learn RandomForest
- Model serialization using joblib
- Performance evaluation with standard metrics

## API Endpoints
- POST /predict - Single prediction (JSON)
- POST /predict/batch - Batch predictions (CSV/JSON)
- GET /health - Service status check

## Project Structure
```
├── backend/         # FastAPI application
├── frontend/        # Streamlit interface
├── data/           # Dataset storage
├── notebooks/      # Analysis notebooks
└── models/         # Saved model files
    └── model.pkl
```