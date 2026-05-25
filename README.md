<div align="center">

# 🩺 Insurance Premium Category Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/marwanthe0/Insurance_Category_predictor)

**A full-stack ML application that predicts insurance premium categories (Low / Medium / High) for both health and car insurance using trained RandomForest models.**

[🚀 Try Live Demo](https://huggingface.co/spaces/marwanthe0/Insurance_Category_predictor) · [📖 Report Bug](https://github.com/Marwanthe0/Insurance_Category_Predictor/issues) · [💡 Request Feature](https://github.com/Marwanthe0/Insurance_Category_Predictor/issues)

</div>

## ✨ Features

- **Dual Insurance Support** — Separate models for health insurance and car insurance
- **Real-time Predictions** — Instant category output (Low / Medium / High)
- **Interactive Analytics Dashboard** — Visual breakdown of prediction factors
- **REST API Backend** — FastAPI with `/predict` and `/predict/batch` endpoints
- **Streamlit Frontend** — Clean, user-friendly web interface
- **Reproducible Pipeline** — Preprocessing + training notebooks included

---

## 🏗️ Architecture

### System Architecture Diagram
```
User Input (Streamlit UI)
           ↓
    ┌──────────────┐
    │  Streamlit   │
    │  Frontend    │
    └──────┬───────┘
           ↓
    ┌──────────────────┐
    │   FastAPI        │
    │   Backend        │
    └──────┬───────────┘
           ↓
    ┌──────────────────────┐
    │  Model Inference     │
    │  ├─ health_model.pkl │
    │  └─ car_model.pkl    │
    └──────┬───────────────┘
           ↓
    Prediction Response
           ↓
    Results + Analytics
```

### Component Overview
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | User interface for predictions & analytics |
| **Backend** | FastAPI | REST API for model inference |
| **Models** | scikit-learn (RandomForest) | ML models for category prediction |
| **Data Processing** | pandas, joblib | Data loading & model serialization |

---

## 🗂️ Project Structure

```
Insurance_Category_Predictor/
│
├── 📄 app.py                      # Streamlit main entry point
├── 📄 Home.py                     # Landing page component
│
├── 📁 pages/                      # Multi-page Streamlit app
│   ├── health_predict.py          # Health insurance prediction UI
│   └── car_predict.py             # Car insurance prediction UI
│
├── 📁 models/                     # Serialized ML models
│   ├── health_model.pkl           # Trained RandomForest (health)
│   └── car_model.pkl              # Trained RandomForest (car)
│
├── 📁 data/                       # Datasets (if applicable)
│   ├── insurance.csv              # Health insurance dataset
│   └── Car_Dataset.csv            # Car insurance dataset
│
├── 📁 notebooks/                  # Training & analysis notebooks
│   ├── ml-model.ipynb             # Health model training pipeline
│   └── car_ml_model.py            # Car model training script
│
├── 📄 requirements.txt            # Python dependencies
├── 📄 main.py                     # FastAPI application (if using API)
├── 📄 README.md                   # This file
└── 📄 .gitignore

```

### Directory Details

#### `/pages/` - Frontend Pages
- **health_predict.py** — Input form + predictions for health insurance
- **car_predict.py** — Input form + predictions for car insurance

#### `/models/` - Serialized Models
- **health_model.pkl** — Pre-trained RandomForest classifier for health insurance
- **car_model.pkl** — Pre-trained RandomForest classifier for car insurance

#### `/data/` - Source Datasets
- **insurance.csv** — Health insurance dataset (training data)
- **Car_Dataset.csv** — Car insurance dataset (training data)

#### `/notebooks/` - Model Training & Development
- **ml-model.ipynb** — Jupyter notebook for health model training & experimentation
- **car_ml_model.py** — Python script for car model training

---

## ⚙️ Installation & Running Locally

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Marwanthe0/Insurance_Category_Predictor.git
cd Insurance_Category_Predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
Visit: **http://localhost:8501**

### 5. (Optional) Run the FastAPI backend separately
```bash
uvicorn main:app --reload
```
API docs at: **http://localhost:8000/docs**

---

## 🤖 Model Details

| Feature | Health Model | Car Model |
|---------|-------------|-----------|
| Algorithm | Random Forest | Random Forest |
| Target Classes | Low / Medium / High | Low / Medium / High |
| Preprocessing | Label encoding, scaling | Label encoding, scaling |
| Serialization | joblib (.pkl) | joblib (.pkl) |

### Key Features Used

**Health Insurance:** Age, BMI, Smoking status, Region, Number of dependents, Sex

**Car Insurance:** Vehicle age, Annual mileage, Previous claims, Vehicle type, Coverage type

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Single prediction (JSON) |
| `POST` | `/predict/batch` | Batch predictions (CSV/JSON) |

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "bmi": 25.4, "smoker": "no", "region": "southwest", "children": 1}'
```

**Example response:**
```json
{
  "prediction": "Medium",
  "confidence": 0.82
}
```

---

## 🙋 Author

**Shafikul Islam Marwan**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/marwanahmed27/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github)](https://github.com/Marwanthe0)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/marwanthe0)

---

<div align="center">
⭐ If you found this useful, please give it a star! It helps a lot.
</div>
